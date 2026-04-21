import logging
import re

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "condition_name",
    "category",
    "subcategory",
    "status",
    "onset",
    "evidence"
]

VALID_STATUS = {"active", "resolved", "suspected"}

# Optional: keep synced with taxonomy top-level categories
VALID_CATEGORIES = {
    "cancer",
    "cardiovascular",
    "infectious",
    "metabolic_endocrine",
    "neurological",
    "pulmonary",
    "gastrointestinal",
    "renal",
    "hematological",
    "immunological",
    "musculoskeletal",
    "toxicological",
    "dental_oral"
}

ONSET_PATTERN = re.compile(r"^[A-Z][a-z]+ \d{4}$")


def validate_condition(cond):
    errors = []

    # required fields
    for field in REQUIRED_FIELDS:
        if field not in cond:
            errors.append(f"Missing field: {field}")

    # stop early if key fields missing
    if errors:
        return errors

    # non-empty string checks
    for field in ["condition_name", "category", "subcategory", "status", "onset"]:
        if not isinstance(cond[field], str) or not cond[field].strip():
            errors.append(f"Field must be non-empty string: {field}")

    # status check
    if cond.get("status") not in VALID_STATUS:
        errors.append(f"Invalid status: {cond.get('status')}")

    # category check
    if cond.get("category") not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {cond.get('category')}")

    # onset check
    if isinstance(cond.get("onset"), str):
        if not ONSET_PATTERN.match(cond["onset"]):
            errors.append(f"Invalid onset format: {cond['onset']}")

    # evidence check
    if not isinstance(cond["evidence"], list):
        errors.append("Evidence must be a list")
    elif len(cond["evidence"]) == 0:
        errors.append("Evidence list cannot be empty")
    else:
        for ev in cond["evidence"]:
            if not isinstance(ev, dict):
                errors.append(f"Evidence must be dict: {ev}")
                continue

            for k in ["note_id", "line_no", "span"]:
                if k not in ev:
                    errors.append(f"Incomplete evidence: missing {k} in {ev}")

            if "note_id" in ev and (not isinstance(ev["note_id"], str) or not ev["note_id"].strip()):
                errors.append(f"Invalid note_id in evidence: {ev}")

            if "line_no" in ev and not isinstance(ev["line_no"], int):
                errors.append(f"line_no must be int in evidence: {ev}")

            if "span" in ev and (not isinstance(ev["span"], str) or not ev["span"].strip()):
                errors.append(f"Invalid span in evidence: {ev}")

    return errors


def validate_output(output):
    all_errors = []

    if "patient_id" not in output:
        all_errors.append("Missing patient_id")

    if "conditions" not in output:
        all_errors.append("Missing conditions list")
        return all_errors

    if not isinstance(output["conditions"], list):
        all_errors.append("conditions must be a list")
        return all_errors

    for idx, cond in enumerate(output["conditions"]):
        errs = validate_condition(cond)
        if errs:
            all_errors.append({
                "condition_index": idx,
                "errors": errs
            })

    return all_errors

if __name__ == "__main__":
    import json

    INPUT_FILE = "mapped_output.json"
    OUTPUT_FILE = "final_output.json"

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            output = json.load(f)

        result = {
                    "patient_id": "patient_06",
                    "conditions": output
                }

        errors = validate_output(result)

        print("\n===== VALIDATION RESULT =====\n")

        if not errors:
            print("No validation errors found.")
        else:
            print(json.dumps(errors, indent=2, ensure_ascii=False))
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error running validator standalone: {e}", exc_info=True)