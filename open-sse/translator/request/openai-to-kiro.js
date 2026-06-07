/**
 * OpenAI to Kiro Request Translator
 * Converts OpenAI Chat Completions format to Kiro/AWS CodeWhisperer format
 */
import { register } from "../index.js";
import { FORMATS } from "../formats.js";
import { v4 as uuidv4 } from "uuid";
import {
  resolveKiroModel,
  isThinkingEnabled,
  buildThinkingSystemPrefix,
  KIRO_AGENTIC_SYSTEM_PROMPT
} from "../../config/kiroConstants.js";


const MAX_KIRO_PAYLOAD_BYTES = 900 * 1024;
const MIN_RECENT_HISTORY_TURNS = 4;
const TOOL_RESULTS_PREFIX = "Tool results:";
const TRUNCATION_PLACEHOLDER = "[Earlier conversation history was truncated to fit Kiro's request limit. Older messages and tool activity were omitted.]";
const CURRENT_MESSAGE_TRUNCATION_MARKER = "\n\n[Middle of current message truncated to fit Kiro's request limit.]\n\n";
const POLLUTED_TOOL_CALL_TEXT = /\[Called tool [^\]]*\]/g;

function sanitizeToolName(name) {
  // Preserve exact tool names so returned OpenAI tool_calls still match the
  // client's registered tool registry. Schema cleanup is safe; renaming tools
  // is not unless we also map Kiro's output name back to the original name.
  const raw = String(name || "tool").trim();
  return raw || "tool";
}

function normalizeToolSchema(schema) {
  const root = schema && typeof schema === "object" && !Array.isArray(schema)
    ? structuredCloneSafe(schema)
    : { type: "object", properties: {} };
  cleanToolSchema(root);
  if (!root.type) root.type = "object";
  if (root.type === "object" && !root.properties) root.properties = {};
  return root;
}

function cleanToolSchema(node) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) {
    node.forEach(cleanToolSchema);
    return;
  }
  delete node.additionalProperties;
  if ("required" in node && (!Array.isArray(node.required) || node.required.length === 0)) {
    delete node.required;
  }
  for (const value of Object.values(node)) cleanToolSchema(value);
}

function structuredCloneSafe(value) {
  try {
    if (typeof structuredClone === "function") return structuredClone(value);
  } catch {}
  try { return JSON.parse(JSON.stringify(value)); } catch { return { type: "object", properties: {} }; }
}

function collectCurrentToolResultIds(currentMessage) {
  const results = currentMessage?.userInputMessage?.userInputMessageContext?.toolResults || [];
  const ids = new Set(results.map(r => r.toolUseId).filter(Boolean));
  return ids.size ? ids : null;
}

function narrateToolResults(toolResults, toolNames) {
  const parts = [];
  for (const result of toolResults || []) {
    const text = (result.content || []).map(c => c.text || "").filter(Boolean).join("\n") || "(no output)";
    const name = toolNames.get(result.toolUseId);
    parts.push(name ? `[${name}] ${text}` : text);
  }
  return parts.length ? `${TOOL_RESULTS_PREFIX}\n\n${parts.join("\n\n")}` : "";
}

function joinText(a, b) {
  a = String(a || "").trim();
  b = String(b || "").trim();
  return a && b ? `${a}\n\n${b}` : (a || b);
}

function sanitizeKiroHistory(history, currentToolResultIds) {
  if (!Array.isArray(history) || history.length === 0) return history || [];

  const toolNames = new Map();
  for (const item of history) {
    for (const toolUse of item.assistantResponseMessage?.toolUses || []) {
      if (toolUse.toolUseId && toolUse.name) toolNames.set(toolUse.toolUseId, toolUse.name);
    }
  }

  let activeAssistantIndex = -1;
  if (currentToolResultIds?.size) {
    const last = history[history.length - 1];
    const uses = last?.assistantResponseMessage?.toolUses || [];
    if (uses.length && uses.every(u => currentToolResultIds.has(u.toolUseId))) {
      activeAssistantIndex = history.length - 1;
    }
  }

  const cleaned = [];
  for (let i = 0; i < history.length; i++) {
    const item = history[i];
    const assistant = item.assistantResponseMessage;
    const user = item.userInputMessage;

    if (assistant) {
      if (assistant.content) {
        assistant.content = assistant.content.replace(POLLUTED_TOOL_CALL_TEXT, "").replace(/\n{3,}/g, "\n\n").trim();
      }
      if (i !== activeAssistantIndex && Array.isArray(assistant.toolUses)) {
        assistant.toolUses = [];
      }
      if (!assistant.content?.trim() && (!assistant.toolUses || assistant.toolUses.length === 0)) {
        continue;
      }
    }

    if (user?.userInputMessageContext) {
      const ctx = user.userInputMessageContext;
      if (ctx.toolResults?.length) {
        user.content = joinText(user.content, narrateToolResults(ctx.toolResults, toolNames));
        delete ctx.toolResults;
      }
      delete ctx.tools;
      if (Object.keys(ctx).length === 0) delete user.userInputMessageContext;
    }

    if (user && !String(user.content || "").trim() && !user.images?.length) user.content = "continue";

    const prev = cleaned[cleaned.length - 1];
    if (user && prev?.userInputMessage && !user.images?.length &&
        String(prev.userInputMessage.content || "").trim() === String(user.content || "").trim()) {
      continue;
    }
    cleaned.push(item);
  }

  while (cleaned[0]?.assistantResponseMessage) cleaned.shift();
  return cleaned;
}

function payloadSize(payload) {
  try { return new TextEncoder().encode(JSON.stringify(payload)).length; } catch { return 0; }
}

function truncateKiroPayload(payload) {
  if (!payload?.conversationState) return;
  let size = payloadSize(payload);
  if (size <= MAX_KIRO_PAYLOAD_BYTES) return;

  const history = payload.conversationState.history || [];
  if (!history.length) return truncateCurrentMessage(payload, size);

  // Drop old history in one coarse pass based on total excess, then do a tiny
  // confirmation loop. This avoids repeatedly stringifying ~1MB payloads once
  // per history item in the common oversized-history case.
  const historyBytes = history.map(item => payloadSize(item));
  let keepFrom = Math.max(0, history.length - MIN_RECENT_HISTORY_TURNS);
  let bytesToDrop = size - MAX_KIRO_PAYLOAD_BYTES;
  for (let i = 0; i < keepFrom && bytesToDrop > 0; i++) {
    bytesToDrop -= historyBytes[i];
    keepFrom = i + 1;
  }

  while (keepFrom < history.length && history[keepFrom]?.assistantResponseMessage) {
    keepFrom++;
  }

  let rebuilt = [placeholderHistory(), ...history.slice(keepFrom)];
  payload.conversationState.history = rebuilt;
  size = payloadSize(payload);

  while (size > MAX_KIRO_PAYLOAD_BYTES && payload.conversationState.history.length > 1) {
    payload.conversationState.history.splice(1, 1);
    while (payload.conversationState.history[1]?.assistantResponseMessage) {
      payload.conversationState.history.splice(1, 1);
    }
    size = payloadSize(payload);
  }

  if (size > MAX_KIRO_PAYLOAD_BYTES) truncateCurrentMessage(payload, size);
}

function placeholderHistory() {
  return {
    userInputMessage: {
      content: TRUNCATION_PLACEHOLDER,
      modelId: "",
      origin: "AI_EDITOR"
    }
  };
}

function truncateCurrentMessage(payload, knownSize = payloadSize(payload)) {
  const msg = payload.conversationState.currentMessage?.userInputMessage;
  if (!msg?.content || knownSize <= MAX_KIRO_PAYLOAD_BYTES) return;

  while (knownSize > MAX_KIRO_PAYLOAD_BYTES && msg.content.length > CURRENT_MESSAGE_TRUNCATION_MARKER.length + 64) {
    const excess = knownSize - MAX_KIRO_PAYLOAD_BYTES;
    const removeChars = Math.min(
      msg.content.length - CURRENT_MESSAGE_TRUNCATION_MARKER.length - 64,
      Math.max(Math.ceil(excess / 2) + 2048, Math.floor(msg.content.length * 0.25))
    );
    const keepChars = msg.content.length - removeChars - CURRENT_MESSAGE_TRUNCATION_MARKER.length;
    const keepStart = Math.max(32, Math.floor(keepChars * 0.35));
    const keepEnd = Math.max(32, keepChars - keepStart);
    msg.content = `${msg.content.slice(0, keepStart)}${CURRENT_MESSAGE_TRUNCATION_MARKER}${msg.content.slice(-keepEnd)}`;
    knownSize = payloadSize(payload);
  }
}

/**
 * Convert OpenAI messages to Kiro format
 * Rules: system/tool/user -> user role, merge consecutive same roles
 */
function convertMessages(messages, tools, model) {
  let history = [];
  let currentMessage = null;
  
  let pendingUserContent = [];
  let pendingAssistantContent = [];
  let pendingToolResults = [];
  let pendingImages = [];
  let currentRole = null;

  // Image support is pre-filtered by caps in translateRequest before reaching here
  const supportsImages = true;

  const flushPending = () => {
    if (currentRole === "user") {
      const content = pendingUserContent.join("\n\n").trim() || "continue";
      const userMsg = {
        userInputMessage: {
          content: content,
          modelId: ""
        }
      };

      // Attach images if present (Kiro API supports images field)
      if (pendingImages.length > 0) {
        userMsg.userInputMessage.images = pendingImages;
      }

      if (pendingToolResults.length > 0) {
        userMsg.userInputMessage.userInputMessageContext = {
          toolResults: pendingToolResults
        };
      }
      
      // Add tools to first user message
      if (tools && tools.length > 0 && history.length === 0) {
        if (!userMsg.userInputMessage.userInputMessageContext) {
          userMsg.userInputMessage.userInputMessageContext = {};
        }
        userMsg.userInputMessage.userInputMessageContext.tools = tools.map(t => {
          const name = t.function?.name || t.name;
          let description = t.function?.description || t.description || "";
          
          if (!description.trim()) {
            description = `Tool: ${name}`;
          }
          
          const schema = t.function?.parameters || t.parameters || t.input_schema || {};
          const cleanedName = sanitizeToolName(name);

          return {
            toolSpecification: {
              name: cleanedName,
              description,
              inputSchema: { json: normalizeToolSchema(schema) }
            }
          };
        });
      }
      
      history.push(userMsg);
      currentMessage = userMsg;
      pendingUserContent = [];
      pendingToolResults = [];
      pendingImages = [];
    } else if (currentRole === "assistant") {
      const content = pendingAssistantContent.join("\n\n").trim() || "...";
      const assistantMsg = {
        assistantResponseMessage: {
          content: content
        }
      };
      history.push(assistantMsg);
      pendingAssistantContent = [];
    }
  };

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    let role = msg.role;
    
    // Normalize: system/tool -> user
    if (role === "system" || role === "tool") {
      role = "user";
    }
    
    // If role changes, flush pending
    if (role !== currentRole && currentRole !== null) {
      flushPending();
    }
    currentRole = role;
    
    if (role === "user") {
      // Extract content
      let content = "";
      if (typeof msg.content === "string") {
        content = msg.content;
      } else if (Array.isArray(msg.content)) {
        const textParts = [];
        for (const c of msg.content) {
          if (c.type === "text" || c.text) {
            textParts.push(c.text || "");
          } else if (supportsImages && c.type === "image_url") {
            // OpenAI format: image_url.url with data URI
            const url = c.image_url?.url || "";
            const base64Match = url.match(/^data:([^;]+);base64,(.+)$/);
            if (base64Match) {
              const mediaType = base64Match[1];
              const format = mediaType.split("/")[1] || mediaType;
              pendingImages.push({ format, source: { bytes: base64Match[2] } });
            } else if (url.startsWith("http://") || url.startsWith("https://")) {
              // Kiro only supports base64 — fallback to URL text
              textParts.push(`[Image: ${url}]`);
            }
          } else if (supportsImages && c.type === "image") {
            // Claude format: source.type = "base64", source.media_type, source.data
            if (c.source?.type === "base64" && c.source?.data) {
              const mediaType = c.source.media_type || "image/png";
              const format = mediaType.split("/")[1] || mediaType;
              pendingImages.push({ format, source: { bytes: c.source.data } });
            }
          }
        }
        content = textParts.join("\n");
        
        // Check for tool_result blocks
        const toolResultBlocks = msg.content.filter(c => c.type === "tool_result");
        if (toolResultBlocks.length > 0) {
          toolResultBlocks.forEach(block => {
            const text = Array.isArray(block.content) 
              ? block.content.map(c => c.text || "").join("\n")
              : (typeof block.content === "string" ? block.content : "");
            
            pendingToolResults.push({
              toolUseId: block.tool_use_id,
              status: "success",
              content: [{ text: text }]
            });
          });
        }
      }
      
      // Handle tool role (from normalized). OpenAI-compatible coding agents may
      // send tool content as an array of text blocks; preserve it instead of
      // silently converting it to an empty Kiro tool result.
      if (msg.role === "tool") {
        const toolContent = typeof msg.content === "string"
          ? msg.content
          : Array.isArray(msg.content)
            ? msg.content.map(c => typeof c === "string" ? c : (c?.text || c?.content || "")).join("\n")
            : (msg.content ? JSON.stringify(msg.content) : "");
        pendingToolResults.push({
          toolUseId: msg.tool_call_id,
          status: "success",
          content: [{ text: toolContent }]
        });
      } else if (content) {
        pendingUserContent.push(content);
      }
    } else if (role === "assistant") {
      // Extract text content and tool uses
      let textContent = "";
      let toolUses = [];
      
      if (Array.isArray(msg.content)) {
        const textBlocks = msg.content.filter(c => c.type === "text");
        textContent = textBlocks.map(b => b.text).join("\n").trim();
        
        const toolUseBlocks = msg.content.filter(c => c.type === "tool_use");
        toolUses = toolUseBlocks;
      } else if (typeof msg.content === "string") {
        textContent = msg.content.trim();
      }
      
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        toolUses = msg.tool_calls;
      }
      
      if (textContent) {
        pendingAssistantContent.push(textContent);
      }
      
      // Store tool uses in last assistant message
      if (toolUses.length > 0) {
        if (pendingAssistantContent.length === 0) {
          // pendingAssistantContent.push("Call tools");
        }
        
        // Flush to create assistant message with toolUses
        flushPending();
        
        const lastMsg = history[history.length - 1];
        if (lastMsg?.assistantResponseMessage) {
          lastMsg.assistantResponseMessage.toolUses = toolUses.map(tc => {
            if (tc.function) {
              return {
                toolUseId: tc.id || uuidv4(),
                name: tc.function.name,
                input: typeof tc.function.arguments === "string" 
                  ? JSON.parse(tc.function.arguments) 
                  : (tc.function.arguments || {})
              };
            } else {
              return {
                toolUseId: tc.id || uuidv4(),
                name: tc.name,
                input: tc.input || {}
              };
            }
          });
        }
        
        currentRole = null;
      }
    }
  }
  
  // Flush remaining
  if (currentRole !== null) {
    flushPending();
  }
  
  // Pop last userInputMessage as currentMessage (search from end, skip trailing assistant messages)
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].userInputMessage) {
      currentMessage = history.splice(i, 1)[0];
      break;
    }
  }

  // Grab tools from first history item BEFORE cleanup removes them
  const firstHistoryTools = history[0]?.userInputMessage?.userInputMessageContext?.tools;

  // Clean up history for Kiro API compatibility
  history.forEach(item => {
    if (item.userInputMessage?.userInputMessageContext?.tools) {
      delete item.userInputMessage.userInputMessageContext.tools;
    }
    if (item.userInputMessage?.userInputMessageContext &&
        Object.keys(item.userInputMessage.userInputMessageContext).length === 0) {
      delete item.userInputMessage.userInputMessageContext;
    }
    if (item.userInputMessage && !item.userInputMessage.modelId) {
      item.userInputMessage.modelId = model;
    }
  });

  history = sanitizeKiroHistory(history, collectCurrentToolResultIds(currentMessage));

  // Merge consecutive user messages (Kiro requires alternating user/assistant)
  const mergedHistory = [];
  for (let i = 0; i < history.length; i++) {
    const current = history[i];
    if (current.userInputMessage &&
        mergedHistory.length > 0 &&
        mergedHistory[mergedHistory.length - 1].userInputMessage) {
      const prev = mergedHistory[mergedHistory.length - 1];
      prev.userInputMessage.content += "\n\n" + current.userInputMessage.content;
    } else {
      mergedHistory.push(current);
    }
  }

  // Inject tools into currentMessage AFTER cleanup
  if (firstHistoryTools && currentMessage?.userInputMessage &&
      !currentMessage.userInputMessage.userInputMessageContext?.tools) {
    if (!currentMessage.userInputMessage.userInputMessageContext) {
      currentMessage.userInputMessage.userInputMessageContext = {};
    }
    currentMessage.userInputMessage.userInputMessageContext.tools = firstHistoryTools;
  }

  return { history: mergedHistory, currentMessage };
}

/**
 * Build Kiro payload from OpenAI format
 *
 * Two 9router-specific behaviours implemented here:
 *
 * 1. `-agentic` model suffix. Synthetic variant — same upstream model, but we
 *    inject a chunked-write system prompt to keep large file writes under
 *    Kiro's 2-3 minute server timeout. The suffix is stripped before being
 *    sent upstream.
 *
 * 2. Thinking / reasoning. Kiro does not accept `thinking.type` or
 *    `reasoning_effort` natively. The only way to enable reasoning is to
 *    inject `<thinking_mode>enabled</thinking_mode>` into the user content
 *    sent upstream. Detection covers Anthropic-Beta header, Claude API
 *    `thinking`, OpenAI `reasoning_effort`, AMP/Cursor magic tags, and model
 *    name hints.
 */
export function buildKiroPayload(model, body, stream, credentials) {
  const messages = body.messages || [];
  const tools = body.tools || [];
  const maxTokens = 32000;
  const temperature = body.temperature;
  const topP = body.top_p;

  const { upstream: upstreamModel, agentic, thinking: modelImpliesThinking } = resolveKiroModel(model);
  const thinkingEnabled = modelImpliesThinking || isThinkingEnabled(body, null, model);

  const { history, currentMessage } = convertMessages(messages, tools, upstreamModel);

  const profileArn = credentials?.providerSpecificData?.profileArn || "";

  let finalContent = currentMessage?.userInputMessage?.content || "";
  const timestamp = new Date().toISOString();

  // Build the system-prompt prefix that goes ABOVE the user message body.
  // Order: thinking_mode tag first (so Kiro sees it before any user text),
  // then context/timestamp marker, then optional agentic chunked-write prompt.
  const prefixParts = [];
  if (thinkingEnabled) {
    prefixParts.push(buildThinkingSystemPrefix());
  }
  prefixParts.push(`[Context: Current time is ${timestamp}]`);
  if (agentic) {
    prefixParts.push(KIRO_AGENTIC_SYSTEM_PROMPT);
  }
  finalContent = `${prefixParts.join("\n\n")}\n\n${finalContent}`;

  const payload = {
    conversationState: {
      chatTriggerType: "MANUAL",
      conversationId: uuidv4(),
      currentMessage: {
        userInputMessage: {
          content: finalContent,
          modelId: upstreamModel,
          origin: "AI_EDITOR",
          ...(currentMessage?.userInputMessage?.images?.length > 0 && {
            images: currentMessage.userInputMessage.images
          }),
          ...(currentMessage?.userInputMessage?.userInputMessageContext && {
            userInputMessageContext: currentMessage.userInputMessage.userInputMessageContext
          })
        }
      },
      history: history
    }
  };

  if (profileArn) {
    payload.profileArn = profileArn;
  }

  if (maxTokens || temperature !== undefined || topP !== undefined) {
    payload.inferenceConfig = {};
    if (maxTokens) payload.inferenceConfig.maxTokens = maxTokens;
    if (temperature !== undefined) payload.inferenceConfig.temperature = temperature;
    if (topP !== undefined) payload.inferenceConfig.topP = topP;
  }

  truncateKiroPayload(payload);

  // Tag payload so the executor can route the upstream model id correctly.
  Object.defineProperty(payload, "_kiroUpstreamModel", {
    value: upstreamModel,
    enumerable: false
  });

  return payload;
}

register(FORMATS.OPENAI, FORMATS.KIRO, buildKiroPayload, null);
