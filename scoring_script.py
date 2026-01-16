import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import f1_score

- name: Check secret is present
  env:
    TEST_LABELS_CSV: ${{ secrets.TEST_LABELS_CSV }}
  run: |
    if [ -z "$TEST_LABELS_CSV" ]; then
      echo "Secret TEST_LABELS_CSV is EMPTY or not set"
      exit 1
    fi
    echo "Secret TEST_LABELS_CSV is present (length: ${#TEST_LABELS_CSV})"
    mkdir -p data
    printf "%s" "$TEST_LABEL_CSV" > data/test_labels.csv
    test -s data/test_labels.csv

TRUTH_PATH = "data/test_labels.csv"


def validate(df, name):
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{name} must contain columns: id, target. Found: {df.columns.tolist()}")

    if df["id"].duplicated().any():
        raise ValueError(f"{name} has duplicate ids. Each id must appear once.")

    # Convert target to int and validate values
    df["target"] = df["target"].astype(int)
    invalid = set(df["target"].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"{name} has invalid target values: {sorted(invalid)}. Allowed: 0,1.")


def score(submission_file, truth_path=TRUTH_PATH):
    sub = pd.read_csv(submission_file)
    truth = pd.read_csv(truth_path)

    validate(sub, "submission")
    validate(truth, "truth")

    merged = truth.merge(sub, on="id", how="inner", suffixes=("_true", "_pred"))
    if merged.empty:
        raise ValueError("No matching IDs between truth and submission.")

    score_val = f1_score(merged["target_true"], merged["target_pred"], average="macro")

    missing = sorted(set(truth["id"]) - set(sub["id"]))
    extra = sorted(set(sub["id"]) - set(truth["id"]))

    return {
        "macro_f1": float(score_val),
        "n_scored": int(len(merged)),
        "n_truth": int(len(truth)),
        "n_submission": int(len(sub)),
        "n_missing_ids": int(len(missing)),
        "n_extra_ids": int(len(extra)),
        "missing_ids_preview": missing[:10],
        "extra_ids_preview": extra[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_file", help="Path to submission CSV")
    parser.add_argument("--truth", default=TRUTH_PATH, help="Path to hidden truth CSV")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    try:
        res = score(args.submission_file, args.truth)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Submission Macro-F1 Score: {res['macro_f1']:.4f}")
    print(f"Scored {res['n_scored']} rows (truth={res['n_truth']}, submission={res['n_submission']}).")

    if res["n_missing_ids"]:
        print(f"Warning: missing {res['n_missing_ids']} ids (preview: {res['missing_ids_preview']})")
    if res["n_extra_ids"]:
        print(f"Warning: extra {res['n_extra_ids']} ids (preview: {res['extra_ids_preview']})")

    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
