# Clinical Condition Extraction Pipeline

A patient-level NLP pipeline that extracts structured medical conditions from longitudinal clinical notes using an OpenAI-compatible LLM API.

For each patient, the system reads all clinical notes chronologically and produces a structured JSON containing every detected condition with its category, subcategory, status, onset date, and supporting evidence spans.

---

## Repository Structure

```
clinical_pipeline/
│
├── main.py                  # Entry point — CLI orchestrator
├── requirements.txt         # Python dependencies
├── taxonomy.json            # Condition taxonomy for category/subcategory mapping
├── README.md
├── patients.txt
├── problem_staments.txt
├── report.pdf              # Design report
├── dev/                    # Testing Dataset
├── labels/                 # GT of Training dataset
├── output/
├── train/
│
└── pipeline/
    ├── loader.py            # Note loading and date extraction
    ├── parser.py            # Section detection and block parsing
    ├── llm_extractor.py     # LLM-based condition extraction
    ├── llm_normalizer.py    # LLM-based name normalization
    ├── aggregator.py        # Cross-note mention merging and status resolution
    ├── mapper.py            # Subcategory mapping via taxonomy
    ├── post_aggregator.py   # Onset formatting and final filtering
    ├── evaluator.py         # Local evaluation against ground truth labels
    ├── validator.py         # Output schema validation
    ├── utils.py             # Shared LLM retry wrapper
    └── logger.py            # Logging setup
```

---

## Pipeline Stages

```
Notes (text_*.md)
     │
     ▼
[1] Loader         — reads all notes, extracts encounter dates, 1-indexed line numbers
     │
     ▼
[2] Parser         — detects sections (Diagnoses, Medical History, Imaging, ...)
                     joins multi-line bullets, splits inline paragraph sections
     │
     ▼
[3] LLM Extractor  — single LLM call per patient
                     returns condition name + status + category + source_id per mention
     │
     ▼
[4] Normalizer     — LLM batch call: standardises synonymous condition names
                     e.g. "Arterial hypertension" → "hypertension"
     │
     ▼
[5] Aggregator     — groups mentions by condition name
                     resolves final status from latest note
                     builds evidence list (note_id, line_no, span)
     │
     ▼
[6] Post-Aggregator — formats onset as "Month YYYY", drops unknown-category conditions
     │
     ▼
[7] Mapper         — LLM call per condition: assigns subcategory from taxonomy
     │
     ▼
[8] Formatter      — selects final 6 fields: name, category, subcategory,
                     status, onset, evidence
     │
     ▼
[9] Validator      — checks schema compliance, logs any errors
     │
     ▼
Output: patient_XX.json
```

---

## Environment Setup

All LLM calls use an OpenAI-compatible API. Set these three variables before running — nothing is hardcoded.

**Linux / macOS**
```bash
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="model_name"
```

**Windows (Command Prompt — current session)**
```cmd
set OPENAI_BASE_URL=https://api.example.com/v1
set OPENAI_API_KEY=your_api_key
set OPENAI_MODEL=model_name
```

**Windows (persistent across sessions)**
```cmd
setx OPENAI_BASE_URL "https://api.example.com/v1"
setx OPENAI_API_KEY "your_api_key"
setx OPENAI_MODEL "model_name"
```
> After `setx`, open a new terminal for the variables to take effect.

**Optional — Gemini API**
```bash
export USE_GEMINI=true
export GEMINI_API_KEY="your_gemini_key"
export OPENAI_MODEL="gemini-model-name"
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Input Format

```
data/
└── dev/
    ├── patient_01/
    │   ├── text_0.md
    │   ├── text_1.md
    │   └── text_2.md
    └── patient_02/
        ├── text_0.md
        └── text_1.md
```

Patient list file (`patients.json`):
```json
["patient_01", "patient_02"]
```

Notes are markdown files named `text_N.md` where N is the chronological index (0 = earliest).

---

## Running the Pipeline

```bash
python main.py \
  --data-dir ./data/dev \
  --patient-list ./patients.json \
  --output-dir ./output
```

### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | required | Path to data directory |
| `--patient-list` | required | Path to JSON list of patient IDs |
| `--output-dir` | required | Directory where output JSONs are written |

---

## Output Format

One JSON file per patient written to `--output-dir`:

```
output/
├── patient_01.json
└── patient_02.json
```

**Schema:**
```json
{
  "patient_id": "patient_01",
  "conditions": [
    {
      "condition_name": "hypertension",
      "category": "cardiovascular",
      "subcategory": "hypertensive",
      "status": "active",
      "onset": "April 2017",
      "evidence": [
        {
          "note_id": "text_0",
          "line_no": 12,
          "span": "Arterial hypertension"
        }
      ]
    }
  ]
}
```

**Field definitions:**

| Field | Description |
|---|---|
| `condition_name` | Normalised clinical name |
| `category` | Top-level taxonomy category (13 options) |
| `subcategory` | Taxonomy subcategory within category |
| `status` | `active`, `resolved`, or `suspected` — reflects latest note |
| `onset` | Earliest documented date in `"Month YYYY"` format, or `null` |
| `evidence` | All supporting spans across all notes |

---

## Taxonomy Categories

| Category | Covers |
|---|---|
| `cancer` | Tumors, malignancies, metastases |
| `cardiovascular` | Heart, vessels, circulatory system |
| `infectious` | Bacterial, viral, fungal, parasitic infections |
| `metabolic_endocrine` | Diabetes, thyroid, hormones, metabolism |
| `neurological` | Brain, spinal cord, peripheral nerves |
| `pulmonary` | Lungs, airways, respiratory |
| `gastrointestinal` | Digestive system, liver, pancreas |
| `renal` | Kidneys, urinary tract |
| `hematological` | Blood cells, coagulation |
| `immunological` | Immune system, autoimmune, allergy |
| `musculoskeletal` | Bones, joints, connective tissue |
| `toxicological` | Poisoning, toxic exposure, drug toxicity |
| `dental_oral` | Teeth, gums, oral cavity |

---

## Local Evaluation (Development Only)

To score predictions against ground truth labels during development:

1. Place label files at `./labels/patient_XX.json`
2. Uncomment the `evaluate_all(...)` block at the bottom of `main.py`
3. Run the pipeline — evaluation results print to console and `logs/pipeline.log`

---

## Design Decisions

- **Single LLM extraction call per patient** — all notes combined into one prompt, reducing total API calls and improving cross-note context
- **LLM normalizer** — handles synonymous names, language differences, and abbreviations before aggregation
- **Latest-note status rule** — final status always reflects the most recent clinical note mentioning the condition
- **Taxonomy-constrained output** — subcategory mapper enforces valid taxonomy keys; fallback to closest valid subcategory prevents silent drops
- **Retry with exponential backoff** — all LLM calls retry up to 5 times with jitter, handling rate limits gracefully
- **No hardcoded endpoints or model names** — fully controlled via environment variables at runtime

---

## Requirements
Install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- Compatible with any OpenAI-style API provider
- Gemini support available via `USE_GEMINI=true` environment variable
- Model name is injected at runtime via `OPENAI_MODEL` — never hardcoded
- All logs written to `logs/pipeline.log`
