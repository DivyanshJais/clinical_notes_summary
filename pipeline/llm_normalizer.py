import os
import requests
import logging
from openai import OpenAI
#from logger import setup_logger
from pipeline.logger import setup_logger
from pipeline.utils import call_llm_with_retry
from google import genai

logger = setup_logger()
import json
CACHE = {}
ALLOWED_CATEGORIES = {
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
    "dental_oral",
    "unknown",
}

def norm_key(s: str) -> str:
    return s.strip().lower()

def safe_json_parse(text):
    import re
    try:
        return json.loads(text)
    except:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []
    
    
def batch_normalize_llm(condition_items):
    """
    Takes list of condition items and returns mapping dict
    """

    condition_items = sorted(
        condition_items,
        key=lambda x: (
            norm_key(x.get("category", "unknown")),
            norm_key(x.get("condition_name", ""))
        )
    )
    
    to_query = []
    for item in condition_items:
        cache_key = f'{norm_key(item["condition_name"])}|||{norm_key(item.get("category", "unknown"))}'
        if cache_key not in CACHE:
            to_query.append(item)

    if not to_query:
        return CACHE

    formatted_inputs = json.dumps(to_query, indent=2, ensure_ascii=False)

    prompt = f"""
You are a clinical normalization system.

You will receive a batch of extracted condition items for ONE patient.

Your job is to normalize them CONSISTENTLY across the whole batch.

IMPORTANT:
Treat this as a global normalization task, not independent row-by-row rewriting.

WORKFLOW YOU MUST FOLLOW:
1. First read ALL input items together.
2. Identify which items refer to the SAME underlying clinical condition.
3. Build an internal canonical condition list for this patient.
4. Then output one result row for EACH input item.
5. If multiple inputs refer to the same disease, they MUST reuse the EXACT SAME normalized_name.
6. Only create a new normalized_name when the condition is truly different.

TASKS:
1. Normalize the condition into a concise medically standard canonical name
2. Verify the extracted category and correct it if needed

STRICT RULES:
- Output exactly one row per input item
- Do NOT skip any input
- Do NOT merge output rows
- "original" must exactly match the input condition string
- "input_category" must exactly match the input category string
- Use the SAME canonical normalized_name for synonymous or near-equivalent disease mentions in this batch
- Preserve true clinical distinctions:
  - primary cancer vs metastasis are different
  - unrelated diseases must remain different
- Do NOT convert procedures, treatments, devices, or symptoms into diseases
- If the input is not a disease, keep the normalized name the same and category as "unknown"

CANONICALIZATION RULES:
- Prefer one medically standard canonical name per disease cluster
- Prefer site-specific names when the site is explicit
- Prefer more specific names over vague names when clearly supported
- Keep metastases separate from primary tumors
- Singular/plural variants should usually map to the same canonical name
- Synonyms should map to the same canonical name

Examples:
- "Arterial hypertension" + cardiovascular -> "hypertension" + cardiovascular
- "Diabetes mellitus type II" + metabolic_endocrine -> "type 2 diabetes mellitus" + metabolic_endocrine
- "Non-insulin-dependent diabetes mellitus type II" + metabolic_endocrine -> "type 2 diabetes mellitus" + metabolic_endocrine
- "Oral thrush" + infectious -> "oral candidiasis" + infectious
- "Hypacusis" + neurological -> "hearing loss" + neurological

BATCH CONSISTENCY EXAMPLES:
- If the batch contains:
  - "tongue carcinoma"
  - "tongue base carcinoma"
  - "base of tongue carcinoma"
  and they refer to the same primary tumor,
  then they should all normalize to the SAME canonical name, for example:
  "base of tongue carcinoma"

- If the batch contains:
  - "metastatic carcinoma"
  - "metastatic squamous cell carcinoma"
  and the more specific metastatic diagnosis is supported,
  then prefer the more specific canonical metastatic name.

Allowed categories:
- cancer: Tumors and cancer-related conditions
- cardiovascular: Conditions affecting the heart, blood vessels, and circulatory system
- infectious: Conditions caused by pathogenic organisms
- metabolic_endocrine: Disorders of metabolism, hormones, and endocrine glands
- neurological: Conditions affecting the brain, spinal cord, and peripheral nervous system
- pulmonary: Conditions affecting the lungs and respiratory system
- gastrointestinal: Conditions affecting the digestive system, including liver and pancreas
- renal: Conditions affecting the kidneys and urinary tract
- hematological: Conditions affecting blood cells, bone marrow, and coagulation
- immunological: Conditions of the immune system
- musculoskeletal: Conditions affecting bones, joints, and connective tissue
- toxicological: Conditions caused by poisoning, toxic exposures, or substance-related harm
- dental_oral: Conditions affecting the teeth, gums, and oral cavity (excluding oral cancers)
- unknown: use only if no category fits

Return ONLY valid JSON list:
[
  {{
    "original": "exact input condition_name",
    "input_category": "exact input category",
    "normalized_name": "single canonical disease name reused consistently across the batch",
    "category": "one_allowed_category"
  }}
]

Input conditions:
{formatted_inputs}
"""

    try:
        response = call_llm_with_retry(prompt, max_retries=5, base_delay=4.0, max_delay=40.0)

        if not response:
            logger.warning("Empty LLM response in normalization")
            return CACHE

        logger.info(f"NORMALIZER RAW OUTPUT:\n{response[:1000]}")
        parsed = safe_json_parse(response)

        if not isinstance(parsed, list):
            logger.error(f"Normalization returned invalid format: {parsed}")
            return CACHE

        for item in parsed:
            if not all(k in item for k in ["original", "normalized_name", "category"]):
                continue

            category = norm_key(item["category"])
            if category not in ALLOWED_CATEGORIES:
                category = "unknown"

            cache_key = f'{norm_key(item["original"])}|||{norm_key(item.get("input_category", "unknown"))}'
            CACHE[cache_key] = {
                "normalized_name": norm_key(item["normalized_name"]),
                "category": category
            }

        for item in to_query:
            cache_key = f'{norm_key(item["condition_name"])}|||{norm_key(item.get("category", "unknown"))}'
            if cache_key not in CACHE:
                logger.warning(f'Missing normalization for: {item}')
                CACHE[cache_key] = {
                    "normalized_name": norm_key(item["condition_name"]),
                    "category": norm_key(item.get("category", "unknown"))
                }

        return CACHE

    except Exception as e:
        logger.error(f"Batch normalization failed: {e}")
        return CACHE