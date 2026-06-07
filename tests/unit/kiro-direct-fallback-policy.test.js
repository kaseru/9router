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

  it("preserves account only for pinned sticky kiro tool sessions on auth failures", () => {
    const base = {
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      error: "unauthorized",
      stickyAccountId: "conn-1",
      connectionId: "conn-1"
    };

    expect(shouldPreserveKiroDirectToolSessionAccount({ ...base, status: 401 })).toBe(true);
    expect(shouldPreserveKiroDirectToolSessionAccount({ ...base, status: 403, error: "forbidden" })).toBe(true);
    expect(shouldPreserveKiroDirectToolSessionAccount({ ...base, stickyAccountId: null, status: 401 })).toBe(false);
    expect(shouldPreserveKiroDirectToolSessionAccount({ ...base, stickyAccountId: "conn-2", status: 401 })).toBe(false);
  });

  it("preserves account only for pinned sticky kiro tool sessions on quota exhaustion", () => {
    const base = {
      provider: "kiro",
      body: { tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }] },
      status: 429,
      error: "usage quota exceeded",
      stickyAccountId: "conn-1",
      connectionId: "conn-1"
    };

    expect(shouldPreserveKiroDirectToolSessionAccount(base)).toBe(true);
    expect(shouldPreserveKiroDirectToolSessionAccount({ ...base, stickyAccountId: null })).toBe(false);
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
