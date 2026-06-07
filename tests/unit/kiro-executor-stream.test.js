import { describe, it, expect } from "vitest";
import { KiroExecutor } from "../../open-sse/executors/kiro.js";

function encodeHeader(name, value) {
  const nameBytes = new TextEncoder().encode(name);
  const valueBytes = new TextEncoder().encode(value);
  const out = new Uint8Array(1 + nameBytes.length + 1 + 2 + valueBytes.length);
  let offset = 0;
  out[offset++] = nameBytes.length;
  out.set(nameBytes, offset); offset += nameBytes.length;
  out[offset++] = 7;
  out[offset++] = (valueBytes.length >> 8) & 0xff;
  out[offset++] = valueBytes.length & 0xff;
  out.set(valueBytes, offset);
  return out;
}

function frame(eventType, payload) {
  const headers = encodeHeader(":event-type", eventType);
  const payloadBytes = new TextEncoder().encode(JSON.stringify(payload));
  const totalLength = 12 + headers.length + payloadBytes.length + 4;
  const out = new Uint8Array(totalLength);
  const view = new DataView(out.buffer);
  view.setUint32(0, totalLength, false);
  view.setUint32(4, headers.length, false);
  // prelude CRC placeholder at 8..11 stays zero; parser does not validate CRC.
  out.set(headers, 12);
  out.set(payloadBytes, 12 + headers.length);
  return out;
}

async function readSse(response) {
  const text = await response.text();
  return text
    .split("\n\n")
    .filter(Boolean)
    .map(line => line.replace(/^data: /, ""));
}

describe("KiroExecutor event stream translation", () => {
  it("keeps argument chunks without ids attached to the current tool call", async () => {
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(frame("toolUseEvent", { toolUseID: "tool_1", toolName: "read_file", input: "{\"path\":" }));
        controller.enqueue(frame("toolUseEvent", { input: "\"/tmp/a\"}", done: true }));
        controller.enqueue(frame("messageStopEvent", {}));
        controller.close();
      }
    });
    const executor = new KiroExecutor();
    const response = executor.transformEventStreamToSSE(new Response(body, { status: 200 }), "kiro-test");
    const chunks = await readSse(response);
    const json = chunks.filter(c => c !== "[DONE]").map(JSON.parse);
    const toolChunks = json.flatMap(c => c.choices[0].delta.tool_calls || []);

    expect(toolChunks[0].id).toBe("tool_1");
    expect(toolChunks[0].function.name).toBe("read_file");
    expect(toolChunks.filter(c => c.id).map(c => c.id)).toEqual(["tool_1"]);
    expect(toolChunks.map(c => c.function?.arguments || "").join("")).toBe("{\"path\":\"/tmp/a\"}");
    expect(json.at(-1).choices[0].finish_reason).toBe("tool_calls");
    expect(chunks.at(-1)).toBe("[DONE]");
  });
});
