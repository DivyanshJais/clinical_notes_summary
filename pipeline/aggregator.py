from collections import defaultdict
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_note_index(note_id):
    """
    Safely extract numeric index from note_id like:
    text_0.md → 0
    text_12 → 12
    """
    m = re.search(r'(\d+)', note_id)
    return int(m.group(1)) if m else 0

def resolve_status(mentions):
    """
    Project rule: final status comes from the latest note mention.
    We trust the extractor's per-mention initial_status.
    """
    mentions_sorted = sorted(
        mentions,
        key=lambda x: extract_note_index(x["note_id"])
    )
    return mentions_sorted[-1]["initial_status"]

def parse_date_safe(date_str):
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

def resolve_onset(mentions):
    explicit_dates = []

    for m in mentions:
        if m.get("stated_date"):
            dt = parse_date_safe(m["stated_date"])
            if dt:
                explicit_dates.append(dt)

    if explicit_dates:
        return min(explicit_dates).strftime("%m/%d/%Y")

    mentions_sorted = sorted(
        mentions,
        key=lambda x: extract_note_index(x["note_id"])
    )

    return mentions_sorted[0].get("note_date", None)


def build_evidence(mentions):
    evidence = []
    seen = set()

    for m in mentions:
        key = (m["note_id"], m["line_no"], m["mention"])
        if key in seen:
            continue
        seen.add(key)

        evidence.append({
            "note_id": m["note_id"],
            "line_no": m["line_no"],
            "span": m["mention"]
        })

    return evidence

INVALID_CONDITION_RE = re.compile(
    r'\b(surgery|procedure|resection|biopsy|dissection|therapy|treatment)\b',
    re.I
)


def aggregate_conditions(mentions):
    """
    Convert mentions -> final condition objects
    """
    grouped = defaultdict(list)

    # group by normalized condition_name
    for m in mentions:
        cond_name = m["condition_name"].lower().strip()
        grouped[cond_name].append(m)

    final_conditions = []

    for cond, group in grouped.items():
        try:
            if INVALID_CONDITION_RE.search(cond):
                continue
            status = resolve_status(group)
            onset = resolve_onset(group)
            evidence = build_evidence(group)

            final_conditions.append({
                "condition_name": cond,
                "status": status,
                "onset": onset,
                "category": group[0].get("category", "unknown"),
                "evidence": evidence
            })

        except Exception as e:
            logger.error(f"Aggregation failed for {cond}: {e}")

    return final_conditions

if __name__ == "__main__":
    import json

    INPUT_FILE = "normalized_output.json"
    OUTPUT_FILE = "aggregated_output.json"

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            mentions = json.load(f)

        final_conditions = aggregate_conditions(mentions)

        print("\n===== AGGREGATED OUTPUT SAMPLE =====\n")
        for c in final_conditions[:20]:
            print(json.dumps(c, indent=2, ensure_ascii=False))

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_conditions, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(final_conditions)} aggregated conditions to {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"Error running aggregator standalone: {e}", exc_info=True)