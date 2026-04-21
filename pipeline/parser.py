import re
from typing import List, Dict, Any, Optional
from pipeline.logger import setup_logger
# from logger import setup_logger

logger = setup_logger()


def _sec(word):
    return re.compile(
        r'^\*{0,2}\s*' + word +
        r'(?:\s+[A-Za-z][A-Za-z\s/\-]*)?\s*:?\s*\*{0,2}\s*:?\s*$',
        re.IGNORECASE
    )


def clean_text(text: str) -> str:
    text = re.sub(r'\*\*+', '', text)          # remove **
    text = re.sub(r'<!--.*?-->', '', text)     # remove HTML comments
    text = re.sub(r'\s+', ' ', text)           # normalize spaces
    return text.strip()


SECTION_PATTERNS = [
    ("Other Diagnoses", _sec(r'other\s+diagnoses?')),
    ("Diagnoses", _sec(r'(diagnoses?|previous\s+diagnoses?(?:\s+and\s+therapies?)?)')),
    ("Medical History", _sec(r'(medical\s+history|history|patient\s+history\s+update)')),
    ("Lab Results", _sec(r'(lab\s+results?|lab\s+values(?:\s+upon\s+discharge)?)')),
    ("Imaging", _sec(r'(imaging|radiology|radiology/nuclear medicine|pet\s*[/\-]?\s*ct|pet\s*ct\s*report|mri(?:\s+brain\s+report)?|ct(?:\s+thorax/abdomen/pelvis\s*\+\s*contrast)?)')),
    ("Histology", _sec(r'(histology|dermatohistology|histology\s+dermatohistology|macroscopy|macroscopic\s+description|microscopic\s+description|microscopic\s+examination|gross\s+examination|gross\s+description)')),
    ("Current Presentation", _sec(r'(current\s+presentation|clinical\s+findings?)')),
    ("Physical Examination", _sec(r'physical\s+examination')),
    ("Therapy and Progression", _sec(r'(therapy|treatment)\s+and\s+progression')),
    ("Therapy and Progression", _sec(r'(assessment|assessment/recommendations|recommendations|discussion|summary|oncology\s+status|dermatological\s+assessment|general\s+status)')),
    ("Surgery", _sec(r'(surgery|operation|procedure|surgery\s+report|operation\s+report|type\s+of\s+surgery)')),
]


PROCEDURE_RE = re.compile(
    r'\b(placement|insertion|surgery|catheter|tracheostomy|peg|shunt)\b', re.I
)


def is_procedure_bullet(text: str) -> bool:
    return bool(PROCEDURE_RE.search(text))


def detect_section(text: str) -> Optional[str]:
    clean = text.strip().lower()

    # 1. Explicit headers
    for name, pattern in SECTION_PATTERNS:
        if pattern.match(clean):
            return name

    # 2. High-confidence phrase detection
    if "surgery report" in clean or "operation report" in clean:
        return "Surgery"

    if re.search(r'\b(histology|dermatohistology|biopsy|microscopy|macroscopy)\b', clean, re.I):
        return "Histology"

    # Findings usually belongs to an already active parent section
    if re.search(r'^\*{0,2}\s*findings\b', clean, re.I):
        return None

    # 3. Structured lab signals
    if re.search(r'parameter|result|reference|complete blood count|liver function tests|electrolytes', clean, re.I):
        return "Lab Results"

    # 4. Imaging / radiology-like report titles
    if re.search(r'\b(ct|mri|x-ray|ultrasound|scan|pet/?ct|pet-ct|radiology|nuclear medicine)\b', clean, re.I):
        return "Imaging"

    # 5. Physical exam
    if re.search(r'\bphysical examination|patient in stable\b', clean, re.I):
        return "Physical Examination"

    # 6. Clinical findings / presentation-like
    if re.search(r'\bclinical findings\b', clean, re.I):
        return "Current Presentation"

    # 7. Assessment / discussion / summary-like sections
    if re.search(r'\b(assessment|recommendations|discussion|summary|oncology status|dermatological assessment|general status)\b', clean, re.I):
        return "Therapy and Progression"

    return None


def split_inline_sections(text):
    pattern = re.compile(r'\*\*\s*([A-Za-z\s]+?)\s*\*\*:?', re.I)

    matches = list(pattern.finditer(text))

    if not matches:
        return [(None, text.strip())]

    result = []

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        detected = detect_section(section_name)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end].strip()

        if content:
            result.append((detected, content))

    return result


def parse_note_sections(note: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines = note["lines"]

    parsed = []
    current_section = "Header"

    i = 0
    while i < len(lines):
        line = lines[i]
        text = line["text"]

        # Detect section
        detected = detect_section(text)

        if detected:
            current_section = detected

            # Remove only leading markdown/header marker, keep useful trailing text
            text = re.sub(r'^\*{0,2}\s*', '', text).strip()

        # ---------- BULLET HANDLING ----------
        bullet_match = re.match(r'^\s*-\s+(.*)', text)

        if bullet_match:
            parts = [bullet_match.group(1).strip()]
            start_line = line["line_no"]

            j = i + 1
            while j < len(lines):
                nxt = lines[j]["text"]

                if (not nxt.strip().startswith("-")) and nxt.strip() and not detect_section(nxt):
                    parts.append(nxt.strip())
                    j += 1
                else:
                    break

            parsed.append({
                "line_no": start_line,
                "end_line_no": lines[j - 1]["line_no"],
                "text": clean_text(" ".join(parts)),
                "section": current_section,
                "is_bullet": True,
                "bullet_text": clean_text(" ".join(parts))
            })

            i = j
            continue

        # ---------- PARAGRAPH BLOCK HANDLING ----------
        if text.strip():
            detected_inline = detect_section(text)
            if detected_inline and detected_inline != current_section:
                current_section = detected_inline

            parts = [text.strip()]
            start_line = line["line_no"]

            j = i + 1
            while j < len(lines):
                nxt = lines[j]["text"]

                if detect_section(nxt) or nxt.strip().startswith("-"):
                    break

                if nxt.strip():
                    parts.append(nxt.strip())

                j += 1

            full_text = " ".join(parts)

            # Split inline bolded section markers if present
            segments = split_inline_sections(full_text)

            for sec, seg_text in segments:
                parsed.append({
                    "line_no": start_line,
                    "end_line_no": lines[j - 1]["line_no"],
                    "text": clean_text(seg_text),
                    "section": sec or current_section,
                    "is_bullet": False,
                    "bullet_text": clean_text(seg_text)
                })

            i = j
            continue

        i += 1

    return parsed


def get_bullets(parsed: List[Dict], section: str) -> List[Dict]:
    return [
        x for x in parsed
        if x["section"] == section and x["is_bullet"]
    ]


def get_section_lines(parsed: List[Dict], section: str) -> List[Dict]:
    return [x for x in parsed if x["section"] == section]


import json
import os

if __name__ == "__main__":
    # ===== CONFIG =====
    INPUT_FILE = "loader_output.json"
    OUTPUT_FILE = "parsed_sections.json"

    try:
        # ===== LOAD INPUT =====
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        final_output = {}

        # ===== PROCESS EACH PATIENT =====
        for patient_id, notes in data.items():
            parsed_notes = []

            for note in notes:
                parsed = parse_note_sections(note)

                parsed_notes.append({
                    "note_id": note["note_id"],
                    "date": note["date"],
                    "parsed_blocks": parsed
                })

            final_output[patient_id] = parsed_notes

        # ===== PRINT OUTPUT =====
        print(json.dumps(final_output, indent=2, ensure_ascii=False))

        # ===== SAVE OUTPUT =====
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Parsed output saved to: {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"Error running section parser: {e}", exc_info=True)