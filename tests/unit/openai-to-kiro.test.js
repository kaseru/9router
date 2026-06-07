/**
 * Unit tests for open-sse/translator/request/openai-to-kiro.js
 *
 * Tests cover:
 *  - buildKiroPayload() - basic message conversion
 *  - Image forwarding fix: images in currentMessage must be included in payload
 */

import { describe, it, expect } from "vitest";
import { buildKiroPayload } from "../../open-sse/translator/request/openai-to-kiro.js";

describe("buildKiroPayload", () => {
  describe("basic message conversion", () => {
    it("should convert a simple text message", () => {
      const body = {
        messages: [{ role: "user", content: "Hello" }]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      expect(currentMsg.userInputMessage.content).toContain("Hello");
      expect(currentMsg.userInputMessage.modelId).toBe("claude-sonnet-4.6");
      expect(currentMsg.userInputMessage.origin).toBe("AI_EDITOR");
    });

    it("should not include images field when no images are present", () => {
      const body = {
        messages: [{ role: "user", content: "No images here" }]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      expect(currentMsg.userInputMessage.images).toBeUndefined();
    });
  });

  describe("image forwarding", () => {
    it("should forward base64 image from image_url content part", () => {
      const fakeBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const body = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Describe this image" },
              { type: "image_url", image_url: { url: `data:image/png;base64,${fakeBase64}` } }
            ]
          }
        ]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      expect(currentMsg.userInputMessage.images).toBeDefined();
      expect(currentMsg.userInputMessage.images).toHaveLength(1);
      expect(currentMsg.userInputMessage.images[0].format).toBe("png");
      expect(currentMsg.userInputMessage.images[0].source.bytes).toBe(fakeBase64);
    });

    it("should forward multiple base64 images", () => {
      const fakeBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const body = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Compare these images" },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${fakeBase64}` } },
              { type: "image_url", image_url: { url: `data:image/png;base64,${fakeBase64}` } }
            ]
          }
        ]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      expect(currentMsg.userInputMessage.images).toHaveLength(2);
      expect(currentMsg.userInputMessage.images[0].format).toBe("jpeg");
      expect(currentMsg.userInputMessage.images[1].format).toBe("png");
    });

    it("should not include images field when images array is empty", () => {
      const body = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Just text" }
            ]
          }
        ]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      expect(currentMsg.userInputMessage.images).toBeUndefined();
    });

    it("should include both images and text content together", () => {
      const fakeBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const body = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is in this image?" },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${fakeBase64}` } }
            ]
          }
        ]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      expect(currentMsg.userInputMessage.content).toContain("What is in this image?");
      expect(currentMsg.userInputMessage.images).toHaveLength(1);
    });

    it("should treat http image URLs as text fallback (Kiro only supports base64)", () => {
      const body = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Look at this" },
              { type: "image_url", image_url: { url: "https://example.com/photo.jpg" } }
            ]
          }
        ]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});

      const currentMsg = result.conversationState.currentMessage;
      // HTTP URLs are not supported by Kiro — converted to text placeholder
      expect(currentMsg.userInputMessage.images).toBeUndefined();
      expect(currentMsg.userInputMessage.content).toContain("[Image: https://example.com/photo.jpg]");
    });
  });

  describe("tool results", () => {
    it("should preserve OpenAI tool message array content", () => {
      const body = {
        messages: [
          { role: "user", content: "Run tool" },
          {
            role: "assistant",
            content: [],
            tool_calls: [{ id: "call_1", type: "function", function: { name: "echo", arguments: "{}" } }]
          },
          {
            role: "tool",
            tool_call_id: "call_1",
            content: [{ type: "text", text: "TOOL_ARRAY_OUTPUT" }]
          },
          { role: "user", content: "What was output?" }
        ],
        tools: [{ type: "function", function: { name: "echo", description: "echo", parameters: { type: "object", properties: {} } } }]
      };

      const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});
      const toolResults = result.conversationState.currentMessage.userInputMessage.userInputMessageContext.toolResults;

      expect(toolResults).toHaveLength(1);
      expect(toolResults[0].toolUseId).toBe("call_1");
      expect(toolResults[0].content[0].text).toBe("TOOL_ARRAY_OUTPUT");
    });
  });
});

describe("Kiro agent history hardening", () => {
  it("flattens old tool results into user history and keeps only current active tool result structured", () => {
    const body = {
      messages: [
        { role: "user", content: "start" },
        { role: "assistant", content: "", tool_calls: [{ id: "old_1", type: "function", function: { name: "exec_command", arguments: "{}" } }] },
        { role: "tool", tool_call_id: "old_1", content: "OLD_OUTPUT" },
        { role: "user", content: "continue" },
        { role: "assistant", content: "", tool_calls: [{ id: "active_1", type: "function", function: { name: "read_file", arguments: "{}" } }] },
        { role: "tool", tool_call_id: "active_1", content: "ACTIVE_OUTPUT" },
      ],
      tools: [{ type: "function", function: { name: "exec_command", description: "exec", parameters: { type: "object" } } }]
    };

    const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});
    const history = result.conversationState.history;
    const oldText = JSON.stringify(history);
    expect(oldText).toContain("Tool results:");
    expect(oldText).toContain("OLD_OUTPUT");
    expect(oldText).not.toContain("[Called tool");

    const currentCtx = result.conversationState.currentMessage.userInputMessage.userInputMessageContext;
    expect(currentCtx.toolResults).toHaveLength(1);
    expect(currentCtx.toolResults[0].toolUseId).toBe("active_1");
  });

  it("removes replayed assistant tool-call narration from history", () => {
    const body = {
      messages: [
        { role: "user", content: "start" },
        { role: "assistant", content: "Let me check.\n\n[Called tool exec_command with input {\"cmd\":\"pwd\"}]" },
        { role: "user", content: "continue" },
      ]
    };

    const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});
    const serialized = JSON.stringify(result.conversationState.history);
    expect(serialized).toContain("Let me check.");
    expect(serialized).not.toContain("[Called tool");
  });

  it("cleans tool schema fields Kiro rejects", () => {
    const body = {
      messages: [{ role: "user", content: "hi" }],
      tools: [{ type: "function", function: {
        name: "tool_with_schema",
        description: "",
        parameters: { type: "object", properties: { x: { type: "string" } }, required: [], additionalProperties: false }
      }}]
    };

    const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});
    const schema = result.conversationState.currentMessage.userInputMessage.userInputMessageContext.tools[0].toolSpecification.inputSchema.json;
    expect(schema.additionalProperties).toBeUndefined();
    expect(schema.required).toBeUndefined();
    expect(schema.type).toBe("object");
  });

  it("truncates oversized Kiro payloads before sending upstream", () => {
    const huge = "x".repeat(12000);
    const messages = [{ role: "user", content: "start" }];
    for (let i = 0; i < 120; i++) {
      messages.push({ role: "assistant", content: `assistant ${i}` });
      messages.push({ role: "user", content: `${huge}-${i}` });
    }
    messages.push({ role: "user", content: "final" });

    const result = buildKiroPayload("claude-sonnet-4.6", { messages }, true, {});
    expect(new TextEncoder().encode(JSON.stringify(result)).length).toBeLessThanOrEqual(900 * 1024);
    expect(JSON.stringify(result.conversationState.history)).toContain("Earlier conversation history was truncated");
  });

  it("preserves exact tool names so client tool registry still matches", () => {
    const longName = "mcp__filesystem__read_file_with_a_very_long_registered_tool_name_that_must_not_change";
    const body = {
      messages: [{ role: "user", content: "hi" }],
      tools: [{ type: "function", function: { name: longName, description: "read", parameters: { type: "object" } } }]
    };

    const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});
    const spec = result.conversationState.currentMessage.userInputMessage.userInputMessageContext.tools[0].toolSpecification;
    expect(spec.name).toBe(longName);
  });

  it("truncates current message middle while preserving prefix and latest user tail", () => {
    const head = "IMPORTANT_PREFIX_KEEP";
    const tail = "IMPORTANT_LATEST_TAIL_KEEP";
    const body = {
      messages: [{ role: "user", content: `${head}\n${"x".repeat(1100 * 1024)}\n${tail}` }]
    };

    const result = buildKiroPayload("claude-sonnet-4.6", body, true, {});
    const content = result.conversationState.currentMessage.userInputMessage.content;
    expect(new TextEncoder().encode(JSON.stringify(result)).length).toBeLessThanOrEqual(900 * 1024);
    expect(content).toContain(head);
    expect(content).toContain(tail);
    expect(content).toContain("Middle of current message truncated");
  });
});
