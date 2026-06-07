import { describe, it, expect } from "vitest";
import { shouldPreserveKiroDirectToolSessionAccount } from "../../src/sse/handlers/chat/kiroDirectFallbackPolicy.js";

describe("shouldPreserveKiroDirectToolSessionAccount", () => {
  it("returns false for non-kiro providers", () => {
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "openai",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 429,
      error: "rate limited"
    })).toBe(false);
  });

  it("returns false for kiro requests without tools", () => {
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "kiro",
      body: { messages: [{ role: "user", content: "hi" }] },
      status: 429,
      error: "rate limited"
    })).toBe(false);
  });

  it("preserves account for kiro tool sessions on auth failures", () => {
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 401,
      error: "unauthorized"
    })).toBe(true);
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 403,
      error: "forbidden"
    })).toBe(true);
  });

  it("preserves account for kiro tool sessions on quota exhaustion", () => {
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 429,
      error: "usage quota exceeded"
    })).toBe(true);
  });

  it("allows fallback for kiro tool sessions on transient rate limit and server errors", () => {
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 429,
      error: "rate limited, retry later"
    })).toBe(false);
    expect(shouldPreserveKiroDirectToolSessionAccount({
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 500,
      error: "upstream crashed"
    })).toBe(false);
  });
});
