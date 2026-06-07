import { requestHasTools } from "../../../../open-sse/utils/sessionFingerprint.js";

function normalizeError(error) {
  return String(error || "").toLowerCase();
}

function isQuotaLike429(status, error) {
  if (status !== 429) return false;
  const text = normalizeError(error);
  return text.includes("quota")
    || text.includes("billing")
    || text.includes("credit")
    || text.includes("insufficient_quota")
    || text.includes("usage_limit")
    || text.includes("exceeded your current quota");
}

export function shouldPreserveKiroDirectToolSessionAccount({ provider, body, status, error, stickyAccountId, connectionId }) {
  if (provider !== "kiro") return false;
  if (!requestHasTools(body)) return false;
  if (!stickyAccountId || stickyAccountId !== connectionId) return false;

  if (status === 401 || status === 403) return true;
  if (isQuotaLike429(status, error)) return true;

  return false;
}
