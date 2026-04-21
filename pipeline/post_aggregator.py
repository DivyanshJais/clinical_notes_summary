from datetime import datetime
from pipeline.logger import setup_logger
#from logger import setup_logger
import json

logger = setup_logger()

def format_onset_month_year(date_str):
    if not date_str:
        return None
    
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            dt=datetime.strptime(date_str, fmt)
            return dt.strftime("%B %Y")
        except:
            continue
    
    return date_str

def run_post_aggregator(conditions):
    final=[]

    for cond in conditions:
        category = cond.get("category", "unknown")

        if category == "unknown":
            continue
        if "onset" in cond:
            cond["onset"] = format_onset_month_year(cond.get("onset"))

        final.append(cond)
    return final

if __name__ == "__main__":
    INPUT_FILE = "aggregated_output.json"
    OUTPUT_FILE = "post_aggregated_output.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        conditions = json.load(f)

    cleaned = run_post_aggregator(conditions)

    print("\n===== POST-AGGREGATION OUTPUT SAMPLE =====\n")
    for c in cleaned[:15]:
        print(json.dumps(c, indent=2, ensure_ascii=False))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(cleaned)} post-aggregated conditions to {OUTPUT_FILE}")