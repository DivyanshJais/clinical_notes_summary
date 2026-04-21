import json
import logging
import re
from collections import defaultdict
from pipeline.utils import call_llm_with_retry

logger = logging.getLogger(__name__)


def load_taxonomy(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_parse_list(text):
    try:
        return json.loads(text)
    except Exception:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return []
        return []


def map_subcategories_batch_llm(category, condition_names, subcategories):
    formatted_subcats = "\n".join(
        [f"- {name}: {desc}" for name, desc in subcategories.items()]
    )

    formatted_conditions = json.dumps(condition_names, indent=2, ensure_ascii=False)

    prompt = f"""
You are a medical classification system.

Task:
For EACH condition below, assign the single best subcategory.

Category: {category}

Allowed subcategories and descriptions:
{formatted_subcats}

Rules:
- Return EXACTLY ONE output row per input condition
- "condition_name" must exactly match the input string
- "subcategory" must be either:
  - one of the allowed subcategories listed above
  - or "unknown" if none of them fit confidently
- Do NOT invent a new subcategory
- Follow the allowed subcategory list strictly
- Choose an allowed subcategory when it is a medically reasonable fit
- Return "unknown" only when the condition is clearly incompatible with all listed subcategories

Return STRICT JSON LIST:
[
  {{
    "condition_name": "exact input condition",
    "subcategory": "one_allowed_subcategory_or_unknown"
  }}
]

Input conditions:
{formatted_conditions}
"""

    response = call_llm_with_retry(prompt, max_retries=5, base_delay=4.0, max_delay=40.0)

    if not response or not response.strip():
        return {name: "unknown" for name in condition_names}

    parsed = safe_json_parse_list(response)
    if not isinstance(parsed, list):
        return {name: "unknown" for name in condition_names}

    result = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue

        name = item.get("condition_name", "")
        subcat = item.get("subcategory", "unknown")

        if name not in condition_names:
            continue

        if subcat not in subcategories and subcat != "unknown":
            subcat = "unknown"

        result[name] = subcat

    for name in condition_names:
        if name not in result:
            result[name] = "unknown"

    return result


def run_mapper(conditions, taxonomy, drop_unknown=True):
    final = []
    grouped = defaultdict(list)

    for cond in conditions:
        category = cond.get("category", "unknown")
        if category == "unknown":
            logger.info(f"Dropping condition with unknown category: {cond.get('condition_name', '')}")
            continue
        grouped[category].append(cond)

    for category, conds in grouped.items():
        subcats = (
            taxonomy.get("condition_categories", {})
            .get(category, {})
            .get("subcategories", {})
        )

        if not subcats:
            for cond in conds:
                logger.info(f"No subcategories for {cond.get('condition_name', '')} | category={category}")
            continue

        condition_names = [c["condition_name"] for c in conds]
        batch_map = map_subcategories_batch_llm(category, condition_names, subcats)

        for cond in conds:
            subcategory = batch_map.get(cond["condition_name"], "unknown")
            cond["subcategory"] = subcategory

            if subcategory == "unknown":
                logger.info(f"Unknown subcategory: {cond['condition_name']} | category={category}")
                if drop_unknown:
                    continue

            final.append(cond)

    return final


def run_final_formatter(conditions):
    return [
        {
            "condition_name": cond.get("condition_name"),
            "category": cond.get("category"),
            "subcategory": cond.get("subcategory"),
            "status": cond.get("status"),
            "onset": cond.get("onset"),
            "evidence": cond.get("evidence", []),
        }
        for cond in conditions
    ]


if __name__ == "__main__":
    INPUT_FILE = "post_aggregated_output.json"
    TAXONOMY_FILE = "taxonomy.json"
    OUTPUT_FILE = "mapped_output.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        conditions = json.load(f)

    taxonomy = load_taxonomy(TAXONOMY_FILE)
    mapped_conditions = run_mapper(conditions, taxonomy)
    mapped_conditions = run_final_formatter(mapped_conditions)

    print("\n===== MAPPED OUTPUT SAMPLE =====\n")
    for c in mapped_conditions[:20]:
        print(json.dumps(c, indent=2, ensure_ascii=False))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapped_conditions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(mapped_conditions)} mapped conditions to {OUTPUT_FILE}")