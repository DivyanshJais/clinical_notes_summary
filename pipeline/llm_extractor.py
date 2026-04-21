import json
import re
import os
from openai import OpenAI
from pipeline.logger import setup_logger
from pipeline.utils import call_llm_with_retry
#from logger import setup_logger
from google import genai

logger = setup_logger()

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []

def normalize_source_id(s):
    s = s.strip().replace(" ", "")
    s = s.replace(":", "::")
    s = re.sub(r'::+', '::', s)
    return s

PROCEDURE_CONDITION_RE = re.compile(
    r'\b(surgery|treatment|procedure|dissection|resection|biopsy|sampling|tracheotomy|tonsillectomy)\b',
    re.I
)

def is_invalid_extracted_condition(name: str, mention: str) -> bool:
    if PROCEDURE_CONDITION_RE.search(name):
        return True
    if mention.lower().startswith("status post") and not re.search(r'\b(cancer|carcinoma|diabetes|hypertension|hypothyroidism|cirrhosis|thrombocytopenia)\b', name, re.I):
        return True
    return False

def extract_stated_date(text: str):
    match = re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
    return match.group(0) if match else None

def extract_conditions_llm(text, parsed):
    prompt = f"""
You are a clinical information extraction system.

Each input line is formatted as:
source_id||[SECTION] text

Your task is to extract clinically meaningful medical CONDITIONS from the text.

A condition is a disease, disorder, diagnosis, malignancy, metastasis, infection, syndrome, chronic illness, complication, or other clinically recognized condition.

The patient population may contain many different diseases across the full taxonomy, so do NOT limit extraction to a few examples.

Important:
- Conditions may appear in structured sections OR in narrative prose.
- A condition listed in [Diagnoses] in one note may appear in [Other Diagnoses], [Medical History], [Current Presentation], [Imaging], or [Histology] in another note.
- Some valid conditions appear only in narrative text.

--------------------------------
ALLOWED CATEGORIES
--------------------------------
Assign EXACTLY ONE category from this list for every extracted condition.

- cancer: Tumors and cancer-related conditions
- cardiovascular: Conditions affecting the heart, blood vessels, and circulatory system
- infectious: Conditions caused by pathogenic organisms. Categorize by organism type, not by affected organ.
- metabolic_endocrine: Disorders of metabolism, hormones, and endocrine glands
- neurological: Conditions affecting the brain, spinal cord, and peripheral nervous system
- pulmonary: Conditions affecting the lungs and respiratory system
- gastrointestinal: Conditions affecting the digestive system, including liver and pancreas
- renal: Conditions affecting the kidneys and urinary tract
- hematological: Conditions affecting blood cells, bone marrow, and coagulation. All low blood cell counts belong here regardless of cause.
- immunological: Conditions of the immune system
- musculoskeletal: Conditions affecting bones, joints, and connective tissue
- toxicological: Conditions caused by poisoning, toxic exposures, or environmental hazards
- dental_oral: Conditions affecting the teeth, gums, and oral cavity (excluding oral cancers)

--------------------------------
SECTIONS TO TREAT AS HIGH VALUE
--------------------------------

1. [Diagnoses] and [Other Diagnoses]
- If a disease or diagnosis is explicitly named here, extract it.
- These sections are strong evidence of a real condition.
- Do NOT drop such items unless they are clearly only procedures, devices, or treatments.

2. [Medical History]
- Extract explicitly stated past or chronic diseases.
- If the wording indicates past-only context, status may be resolved.

3. [Current Presentation]
- Extract explicitly stated diseases, malignancies, complications, or previously known conditions.
- Do NOT extract symptoms alone.

4. [Imaging]
- Extract named diseases or clinically meaningful disease-level abnormalities explicitly stated in imaging text.
- Examples that may be valid: metastasis, cirrhosis, cardiomegaly, pleural effusion, aortic sclerosis, hepatosplenomegaly, renal cysts.
- Do NOT extract vague findings such as “lesion”, “mass”, or “abnormality” unless a recognized disease is explicitly stated.

5. [Histology]
- Extract definitive disease diagnoses or disease-level pathology.
- Extract malignancies, metastases, dysplasia/pre-malignant disease when explicitly stated as disease entities.
- Do NOT extract microscopic descriptions, tissue features, or cellular morphology unless they themselves are a recognized disease entity.

--------------------------------
EXTRACT
--------------------------------
Extract:
- confirmed diseases and diagnoses
- malignancies and metastases
- chronic diseases
- infections
- syndromes
- clinically meaningful complications
- pathology-confirmed diseases
- radiology-confirmed disease-level abnormalities
- historical diseases when explicitly documented
- suspected conditions if uncertainty language is present
- lab-defined disorders ONLY when the text explicitly states the disorder name
  (example: extract “anemia”, but do NOT infer anemia from a raw hemoglobin number alone)

Examples of valid conditions:
- hypertension
- diabetes mellitus
- hypothyroidism
- cirrhosis
- candidiasis
- squamous cell carcinoma
- metastatic carcinoma
- obstructive sleep apnea
- thrombocytopenia
- anemia
- cardiomegaly
- hepatosplenomegaly
- aortic sclerosis
- pleural effusion
- atherosclerosis
- renal cysts
- coagulopathy
- deep vein thrombosis
- bacteremia
- ventriculitis
- subdural hematoma
- cerebral edema

--------------------------------
DO NOT EXTRACT
--------------------------------
Do NOT extract:
- symptoms only
- procedures or treatments
- anatomical sites alone
- raw lab values or measurements alone
- pathological features that are not diseases
- generic findings like “mass”, “lesion”, or “finding” unless a disease is explicitly stated
- devices, tubes, drains, catheters
- operative or procedural narrative unless a disease is explicitly stated

Examples of things to NOT extract are pain, swelling, fever, nausea, vomiting, dysphagia, odynophagia, resection, biopsy, tracheostomy, lymph node dissection, catheter, elevated CRP, pleomorphism, dyskeratosis, cellular atypia, surgical approach details

--------------------------------
SPECIAL RULES
--------------------------------

1. STATUS POST / HISTORY OF
- If a line contains “status post”, “s/p”, or “history of”:
  - extract the condition ONLY if a disease is explicitly named
  - do NOT extract if it refers only to a procedure

2. NEGATION
- Do NOT extract negated conditions unless they represent meaningful past disease context.
Examples:
- “no evidence of metastases” -> do NOT extract metastases
- “history of colon cancer, now no evidence of disease” -> extract colon cancer with resolved status

3. DUPLICATION
- If the same condition appears in multiple different lines or notes, extract EACH occurrence with its own source_id.
- Do not collapse duplicates here.

4. NO GUESSING
- Do NOT invent diagnoses.
- Do NOT infer a disease that is not explicitly supported by the text.
- But if the text explicitly names a recognized condition in narrative prose, extract it.

--------------------------------
STATUS ASSIGNMENT
--------------------------------
Assign exactly one of:
- active
- resolved
- suspected

ACTIVE:
Confirmed and currently present — whether newly diagnosed, chronic, worsening, recurrent, or under current management.

RESOLVED:
No longer present, or documented only as past disease context.

SUSPECTED:
Not yet diagnostically confirmed.

Important:
- If a condition appears in [Diagnoses] or [Other Diagnoses] with no uncertainty language, prefer active.
- Do NOT mark a condition as resolved just because it is old or chronic.
- Use resolved only when the text clearly indicates past-only / remission / no longer present / historical context.
- If uncertainty language is present, prefer suspected.

--------------------------------
OUTPUT REQUIREMENTS
--------------------------------
Return ONLY a JSON array.
Do NOT include markdown.
Do NOT include explanations.

Each item must be:
[
  {{
    "condition_name": "concise medically standard name",
    "status": "active|resolved|suspected",
    "category": "one_allowed_category",
    "source_id": "text_1::12"
  }}
]

--------------------------------
TEXT
--------------------------------
{text}
"""

    try:
        logger.info("Calling LLM for extraction...")
        logger.info(f"LLM input length: {len(text)} characters")
        content = call_llm_with_retry(prompt, max_retries=5, base_delay=4.0, max_delay=40.0)

        if not content or not content.strip():
            logger.warning("Empty LLM response")
            return []

        content = content.strip()

        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "").strip()

        try:
            data = safe_json_parse(content)

            if not isinstance(data, list):
                logger.error("Parsed data is not list")
                return []
        except Exception:
            logger.error(f"JSON parse failed. Raw output: {content[:300]}")
            return []

        logger.info(f"Extracted {len(data)} conditions from LLM")

        line_map = {
            f'{p["note_id"]}::{p["line_no"]}': p
            for p in parsed
        }

        mentions = []
        for d in data:
            source_id = normalize_source_id(d.get("source_id", ""))

            if source_id not in line_map:
                logger.warning(f"Unmatched source_id: {source_id}")
                continue

            p = line_map[source_id]
            candidate_name = d["condition_name"].strip()
            # if is_invalid_extracted_condition(candidate_name, p["text"]):
            #     continue

            mentions.append({
                "mention": p["text"],
                "condition_name": d["condition_name"],
                "initial_status": d["status"],
                "category": d.get("category", "unknown"),
                "line_no": p["line_no"],
                "note_id": p["note_id"],
                "note_date": p.get("note_date"),
                "stated_date": extract_stated_date(p["text"]),
                "section": p["section"]
            })

        seen = set()
        unique_mentions = []

        for m in mentions:
            key = (m["condition_name"].lower(), m["line_no"], m["note_id"])
            if key not in seen:
                seen.add(key)
                unique_mentions.append(m)

        return unique_mentions

    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return []


def build_llm_input(parsed):
    TARGET_SECTIONS = [
        "Diagnoses",
        "Other Diagnoses",
        "Medical History",
        "Current Presentation",
        "Imaging",
        "Histology",
        "Therapy and Progression",
        "Findings"
    ]

    lines = []
    for p in parsed:
        if p["section"] in TARGET_SECTIONS:
            text = (p.get("text") or "").strip()
            if not text:
                continue

            source_id = f'{p["note_id"]}::{p["line_no"]}'
            lines.append(f"{source_id}||[{p['section']}] {text}")

    return "\n".join(lines)

if __name__ == "__main__":
    import json
    import time

    with open("parsed_sections.json", "r") as f:
        data = json.load(f)

    all_mentions = []

    for patient_id, notes in data.items():
        print(f"\n========== PATIENT: {patient_id} ==========\n")

        patient_parsed = []

        # Collect all parsed blocks for this patient
        for note in notes:
            parsed = note["parsed_blocks"]

            for p in parsed:
                p["note_id"] = note["note_id"]
                p["note_date"] = note["date"]

            patient_parsed.extend(parsed)

        # Build one LLM input for full patient
        llm_input = build_llm_input(patient_parsed)

        if not llm_input.strip():
            print(f"Skipping patient {patient_id} - no target section content")
            continue

        print(f"Patient {patient_id} | llm_chars={len(llm_input)}")

        print("\n===== LLM INPUT PREVIEW =====\n")
        print(llm_input[:1500])

        mentions = extract_conditions_llm(llm_input, patient_parsed)

        print("\n===== LLM OUTPUT =====\n")
        for m in mentions:
            print(m)

        all_mentions.extend(mentions)

        # Optional small pause for rate limits
        time.sleep(2)

    with open("llm_output.json", "w") as f:
        json.dump(all_mentions, f, indent=2)

    print(f"\nSaved {len(all_mentions)} mentions")