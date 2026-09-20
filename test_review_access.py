import hashlib
import unittest
from review_access import verify_review_code


class ReviewAccessTests(unittest.TestCase):
    def test_disabled_without_hash(self):
        self.assertFalse(verify_review_code("any-code", "")["reviewAccess"])

    def test_invalid_hash_and_wrong_code_deny_access(self):
        self.assertFalse(verify_review_code("code", "z" * 64)["reviewAccess"])
        self.assertFalse(verify_review_code("wrong", hashlib.sha256(b"correct").hexdigest())["reviewAccess"])

    def test_valid_code_grants_bounded_access_without_echoing_code(self):
        result = verify_review_code("  test-only-secret  ", hashlib.sha256(b"test-only-secret").hexdigest(), now=1000)
        self.assertEqual(result, {"reviewAccess": True, "validUntilMillis": 605800000})


if __name__ == "__main__":
    unittest.main()
