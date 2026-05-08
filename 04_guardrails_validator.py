import re
import json
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import PassResult, FailResult

# ── 1. Validator A: PII Detector ──────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    def validate(self, value: str, metadata: dict) -> Validator:
        patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}",
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}"
        }
        found = []
        redacted_value = value
        for name, pattern in patterns.items():
            if re.search(pattern, value):
                found.append(name)
                redacted_value = re.sub(pattern, "[REDACTED]", redacted_value)
        if found:
            return FailResult(error_message=f"PII detected", fix_value=redacted_value)
        return PassResult()

# ── 2. Validator B: JSON Formatter ────────────────────────────────────────
@register_validator(name="custom/json-repair", data_type="string")
class JSONRepair(Validator):
    def validate(self, value: str, metadata: dict) -> Validator:
        # Check if valid first
        try:
            json.loads(value)
            return PassResult()
        except:
            pass

        # Attempt repair
        repaired = value.strip()
        repaired = re.sub(r"```json\n?|\n?```", "", repaired).strip()
        repaired = repaired.replace("'", '"')
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

        try:
            parsed = json.loads(repaired)
            # Return FailResult with fix_value to trigger OnFailAction.FIX
            return FailResult(
                error_message="Invalid JSON format, repaired.",
                fix_value=json.dumps(parsed)
            )
        except Exception:
            fallback = json.dumps({"error": "invalid_format", "raw": value})
            return FailResult(error_message="Broken JSON", fix_value=fallback)

# ── 3. Testing ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 4: Guardrails AI Validators")
    print("=" * 60)

    # Test PII
    pii_guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    print("\n--- Testing PII Detector ---")
    for s in ["My email is john@example.com", "SSN: 123-45-6789"]:
        res = pii_guard.validate(s)
        print(f"In : {s}")
        print(f"Out: {res.validated_output}")

    # Test JSON
    json_guard = Guard().use(JSONRepair(on_fail=OnFailAction.FIX))
    print("\n--- Testing JSON Repair ---")
    test_jsons = [
        "```json\n{'name': 'Bach', 'age': 25}\n```",
        '{"list": [1, 2, 3, ], }'
    ]
    for j in test_jsons:
        res = json_guard.validate(j)
        input_display = j.replace('\n', ' ')
        print(f"In : {input_display}")
        print(f"Out: {res.validated_output}")

if __name__ == "__main__":
    main()
