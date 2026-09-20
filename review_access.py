"""Optional review entitlement. Disabled until the operator configures a code hash."""
import hashlib
import hmac
import os
import time
from pathlib import Path


def verify_review_code(code: str, configured_hash: str | None = None, now: float | None = None):
    expected = configured_hash if configured_hash is not None else os.environ.get("WAKEAI_REVIEW_CODE_SHA256")
    if expected is None:
        try:
            expected = Path(__file__).with_name("review-code.sha256").read_text(encoding="ascii")
        except OSError:
            expected = ""
    expected = expected.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        return {"reviewAccess": False, "validUntilMillis": 0}
    candidate = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
    if not hmac.compare_digest(candidate, expected):
        return {"reviewAccess": False, "validUntilMillis": 0}
    current = time.time() if now is None else now
    return {"reviewAccess": True, "validUntilMillis": int((current + 7 * 86400) * 1000)}
