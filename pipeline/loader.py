import os
import re
from datetime import datetime
from pipeline.logger import setup_logger

logger = setup_logger()

def parse_date(date_str):
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%m/%d/%Y")   # KEEP CONSISTENT
        except:
            continue
    return None

def extract_date_from_text(text):
    text_lower = text.lower()

    # remove DOB
    text_no_dob = re.sub(
        r"(born on|dob|date of birth)[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})",
        "",
        text_lower
    )

    # 1. PRIORITY: encounter/admission
    match = re.search(
        r"(encounter|admission|visit)[^\d]*(\d{1,2}/\d{1,2}/\d{2,4})",
        text_no_dob
    )
    if match:
        return parse_date(match.group(2))

    # 2. range
    match = re.search(
        r"from (\d{1,2}/\d{1,2}/\d{2,4}) to (\d{1,2}/\d{1,2}/\d{2,4})",
        text_no_dob
    )
    if match:
        return parse_date(match.group(1))

    # 3. fallback
    dates = re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}", text_no_dob)
    for d in dates:
        parsed = parse_date(d)
        if parsed:
            return parsed

    return None


def read_note(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        lines = []
        full_text = ""
        full_text = " ".join([line.rstrip("\n") for line in raw_lines])
        for idx, line in enumerate(raw_lines):
            clean_line = line.rstrip("\n")

            lines.append({
                "line_no": idx+1,
                "text": clean_line
            })

        date = extract_date_from_text(full_text)

        return {
            "lines": lines,
            "date": date
        }

    except Exception as e:
        logger.error(f"Failed to read note {file_path}: {e}", exc_info=True)
        raise


def load_patient_notes(data_dir, patient_id):
    try:
        patient_path = os.path.join(data_dir, patient_id)

        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"Patient folder not found: {patient_path}")

        files = [
            f for f in os.listdir(patient_path)
            if f.startswith("text_") and f.endswith(".md")
        ]

        if not files:
            logger.warning(f"No notes found for {patient_id}")

        # Sort properly
        files.sort(key=lambda x: int(re.search(r"text_(\d+)", x).group(1)))

        notes = []

        for file in files:
            try:
                file_path = os.path.join(patient_path, file)

                note_data = read_note(file_path)

                notes.append({
                    "note_id": file.replace(".md", ""),
                    "date": note_data["date"],
                    "lines": note_data["lines"]
                })

            except Exception as e:
                logger.error(f"Skipping file {file} due to error: {e}", exc_info=True)
                continue

        return notes

    except Exception as e:
        logger.error(f"Failed loading patient {patient_id}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    data_dir = "./data"   # adjust
    patient_list_path = "./patient_list.json"

    import json

    with open(patient_list_path) as f:
        patient_ids = json.load(f)

    output = {}

    for pid in patient_ids:
        notes = load_patient_notes(data_dir, pid)
        output[pid] = notes

    print(json.dumps(output, indent=2))

    with open("debug_loader_output.json", "w") as f:
        json.dump(output, f, indent=2)