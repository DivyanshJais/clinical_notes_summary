import json
import os
from collections import defaultdict
from pipeline.logger import setup_logger
import re

logger = setup_logger()


def load_ground_truth(labels_dir, patient_id):
    path = os.path.join(labels_dir, f"{patient_id}.json")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        logger.warning(f"No label found for {patient_id}")
        return None


def normalize_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return name.strip()


def condition_match_score(pred_name, truth_name):
    p_name = normalize_name(pred_name)
    t_name = normalize_name(truth_name)

    if p_name == t_name:
        return 4
    if p_name in t_name or t_name in p_name:
        return 3

    overlap = len(set(p_name.split()) & set(t_name.split()))
    if overlap >= 2:
        return 2
    if overlap == 1:
        return 1
    return 0


def match_conditions(pred, truth):
    matched = []
    used_truth = set()
    used_pred = set()

    for pi, p in enumerate(pred):
        best_idx = None
        best_score = 0

        for ti, t in enumerate(truth):
            if ti in used_truth:
                continue

            score = condition_match_score(p["condition_name"], t["condition_name"])
            if score > best_score:
                best_score = score
                best_idx = ti

        if best_idx is not None and best_score >= 2:
            matched.append((pi, best_idx, pred[pi], truth[best_idx], best_score))
            used_truth.add(best_idx)
            used_pred.add(pi)

    return matched, used_pred, used_truth


def evaluate_patient(pred_output, gt_output, debug=False):
    pred_conditions = pred_output["conditions"]
    gt_conditions = gt_output["conditions"]

    matches, used_pred, used_truth = match_conditions(pred_conditions, gt_conditions)

    tp = len(matches)
    fp = len(pred_conditions) - tp
    fn = len(gt_conditions) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    correct_status = 0
    for _, _, p, t, _ in matches:
        if p["status"] == t["status"]:
            correct_status += 1

    status_acc = correct_status / tp if tp else 0

    debug_info = None
    if debug:
        matched_pairs = [
            {
                "predicted": p["condition_name"],
                "truth": t["condition_name"],
                "pred_status": p.get("status"),
                "truth_status": t.get("status"),
                "score": score
            }
            for _, _, p, t, score in matches
        ]

        false_positives = [
            {
                "condition_name": p["condition_name"],
                "status": p.get("status"),
                "category": p.get("category"),
                "subcategory": p.get("subcategory")
            }
            for i, p in enumerate(pred_conditions)
            if i not in used_pred
        ]

        false_negatives = [
            {
                "condition_name": t["condition_name"],
                "status": t.get("status"),
                "category": t.get("category"),
                "subcategory": t.get("subcategory")
            }
            for i, t in enumerate(gt_conditions)
            if i not in used_truth
        ]

        debug_info = {
            "matched_pairs": matched_pairs,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

    result = {
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0,
        "status_accuracy": status_acc,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

    return result, debug_info


def evaluate_all(output_dir, labels_dir, patient_list, debug=False, debug_dir=None):
    results = []

    if isinstance(patient_list, str):
        with open(patient_list, "r", encoding="utf-8") as f:
            patient_ids = json.load(f)
    else:
        patient_ids = patient_list

    logger.info(f"Evaluating patients: {patient_ids}")

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for patient_id in patient_ids:
        pred_path = os.path.join(output_dir, f"{patient_id}.json")

        if not os.path.exists(pred_path):
            logger.warning(f"Missing prediction for {patient_id}")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)

        gt = load_ground_truth(labels_dir, patient_id)

        if not gt:
            continue

        res, dbg = evaluate_patient(pred, gt, debug=debug)
        results.append(res)

        logger.info(f"{patient_id}: {res}")

        if debug and dbg is not None:
            print(f"\n===== DEBUG: {patient_id} =====")
            print("\nMatched pairs:")
            for m in dbg["matched_pairs"]:
                print(
                    f'  PRED="{m["predicted"]}" [{m["pred_status"]}]'
                    f'  <->  GT="{m["truth"]}" [{m["truth_status"]}]'
                    f'  | score={m["score"]}'
                )

            print("\nFalse positives:")
            for fp_item in dbg["false_positives"]:
                print(f'  {fp_item["condition_name"]} | status={fp_item["status"]}')

            print("\nFalse negatives:")
            for fn_item in dbg["false_negatives"]:
                print(f'  {fn_item["condition_name"]} | status={fn_item["status"]}')

            if debug_dir:
                out_path = os.path.join(debug_dir, f"{patient_id}_debug.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(dbg, f, indent=2, ensure_ascii=False)

    if not results:
        logger.warning("No valid evaluations found!")
        return {}

    avg = defaultdict(float)

    for r in results:
        for k in r:
            avg[k] += r[k]

    for k in avg:
        avg[k] /= len(results)

    logger.info(f"\nFINAL METRICS: {dict(avg)}")
    return avg


if __name__ == "__main__":
    OUTPUT_DIR = "outputs"
    LABELS_DIR = "labels"
    PATIENT_LIST = "patient_list.json"
    DEBUG = True
    DEBUG_DIR = "eval_debug"

    try:
        results = evaluate_all(
            output_dir=OUTPUT_DIR,
            labels_dir=LABELS_DIR,
            patient_list=PATIENT_LIST,
            debug=DEBUG,
            debug_dir=DEBUG_DIR
        )

        print("\n===== EVALUATION RESULT =====\n")
        print(json.dumps(results, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Error running evaluator standalone: {e}", exc_info=True)