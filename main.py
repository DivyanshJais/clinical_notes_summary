import argparse
import json
import os

from pipeline.loader import load_patient_notes
from pipeline.logger import setup_logger
from pipeline.parser import parse_note_sections
from pipeline.normalizer import run_normalizer
from pipeline.mapper import run_mapper, load_taxonomy, run_final_formatter
from datetime import datetime
from pipeline.validator import validate_output
from pipeline.evaluator import evaluate_all
from pipeline.aggregator import aggregate_conditions
from pipeline.llm_extractor import extract_conditions_llm, build_llm_input
from pipeline.post_aggregator import run_post_aggregator

logger = setup_logger()


def get_llm_config():
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("OPENAI_MODEL")
    }

llm_config = get_llm_config()

if not all(llm_config.values()):
    logger.warning("LLM not configured -> running in RULE-ONLY mode")
    use_llm = False
else:
    use_llm = True

def parse_args():
    parser = argparse.ArgumentParser(description="Clinical Extraction Pipeline")

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--patient-list", required=True)
    parser.add_argument("--output-dir", required=True)

    return parser.parse_args()


def main():
    try:
        args = parse_args()
        logger.info("Arguments parsed successfully")


        # Load patient list
        try:
            with open(args.patient_list, "r") as f:
                patient_ids = json.load(f)
            logger.info(f"Loaded {len(patient_ids)} patients")
        except Exception as e:
            logger.error(f"Failed to load patient list: {e}")
            raise

        os.makedirs(args.output_dir, exist_ok=True)

        for patient_id in patient_ids:
            try:
                logger.info(f"Processing {patient_id}")

                notes = load_patient_notes(args.data_dir, patient_id)

                logger.info(f"{patient_id}: Loaded {len(notes)} notes")
                all_mentions = []
                note_dates={}
                all_parsed = []
                for note in notes:
                    if note["date"]:
                        try:
                           note_dates[note["note_id"]] = datetime.strptime(
                            note["date"], "%m/%d/%Y"
                        )
                        except:
                            logger.warning(f"Invalid date in {note['note_id']}")
                    parsed = parse_note_sections(note)
                    for p in parsed:
                        p["note_id"] = note["note_id"]
                        p["note_date"] = note["date"]

                    all_parsed.extend(parsed)

                #single LLM call
                llm_input = build_llm_input(all_parsed)
                extracted = extract_conditions_llm(llm_input, all_parsed)

                all_mentions.extend(extracted)

                logger.info(f"{patient_id}: {len(all_mentions)} mentions extracted")
                taxonomy = load_taxonomy("taxonomy.json")

                mentions_stage5 = run_normalizer(all_mentions, use_llm=use_llm)

                unique_before = sorted(set(m["condition_name"] for m in all_mentions))
                unique_after = sorted(set(m["condition_name"] for m in mentions_stage5))

                logger.info(f"Extracted mentions count: {len(all_mentions)}")
                logger.info(f"Unique extracted condition names: {len(unique_before)}")
                logger.info(f"Unique normalized condition names: {len(unique_after)}")

                conditions_stage6 = aggregate_conditions(mentions_stage5)
                logger.info(f"After aggregation: {len(conditions_stage6)}")

                conditions_stage7 = run_post_aggregator(conditions_stage6)
                logger.info(f"After post aggregation: {len(conditions_stage7)}")

                final_conditions = run_mapper(conditions_stage7, taxonomy)
                final_conditions = run_final_formatter(final_conditions)

                logger.info(f"Mapper kept {len(final_conditions)} / {len(conditions_stage7)} conditions")
                logger.info(f"Final conditions count: {len(final_conditions)}")
                # Placeholder for next stages
                result = {
                    "patient_id": patient_id,
                    "conditions": final_conditions
                }
                errors = validate_output(result)

                if errors:
                    logger.warning(f"Validation errors for {patient_id}: {errors}")
                
                output_path = os.path.join(args.output_dir, f"{patient_id}.json")
                
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"{patient_id}: Output saved successfully")

            except Exception as e:
                logger.error(f"Error processing {patient_id}: {e}", exc_info=True)
                continue  # move to next patient
        # evaluate_all(
        #     output_dir=args.output_dir,
        #     labels_dir="./labels",
        #     patient_list=args.patient_list,
        #     debug=True,
        #     debug_dir="./eval_debug"
        # )
    except Exception as e:
        logger.critical(f"Fatal error in pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()