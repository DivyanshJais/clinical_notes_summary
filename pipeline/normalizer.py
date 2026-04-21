from collections import defaultdict
import re
from pipeline.llm_normalizer import batch_normalize_llm
import json
#from llm_normalizer import batch_normalize_llm


def normalize_name(name):
    return name.lower().strip()

def needs_llm(group):
    names = set(m["condition_name"].lower() for m in group)

    if len(names) == 1:
        return False  # already consistent

    if len(names) > 3:
        return True   # likely messy cluster

    return True

def group_conditions(mentions):
    groups = defaultdict(list)

    for m in mentions:
        key = normalize_name(m["condition_name"])
        groups[key].append(m)

    return groups


def run_normalizer(mentions, use_llm=True):
    items = []
    seen = set()

    for m in mentions:
        key = (m["condition_name"], m.get("category", "unknown"))
        if key not in seen:
            seen.add(key)
            items.append({
                "condition_name": m["condition_name"],
                "category": m.get("category", "unknown")
            })

    mapping = {}
    if use_llm:
        mapping = batch_normalize_llm(items)
    
    mapping_normalized = {
        k.lower(): v for k, v in mapping.items()
    }

    for m in mentions:
        cache_key = f'{m["condition_name"]}|||{m.get("category", "unknown")}'.lower()

        if cache_key in mapping_normalized:
            m["condition_name"] = mapping_normalized[cache_key]["normalized_name"]
            m["category"] = mapping_normalized[cache_key]["category"]
        else:
            m["category"] = m.get("category", "unknown")

    return mentions

if __name__ == "__main__":

    with open("llm_output.json", "r", encoding="utf-8") as f:
        mentions = json.load(f)

    normalized_mentions = run_normalizer(mentions, use_llm=True)

    print("\n===== NORMALIZED OUTPUT SAMPLE =====\n")
    for m in normalized_mentions[:20]:
        print({
            "mention": m["mention"],
            "condition_name": m["condition_name"],
            "category": m.get("category")
        })

    with open("normalized_output.json", "w", encoding="utf-8") as f:
        json.dump(normalized_mentions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(normalized_mentions)} normalized mentions")