import sys
import pandas as pd
from sklearn.metrics import f1_score

def must_have_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {df.columns.tolist()}")

if __name__ == "__main__":
    submission_file = sys.argv[1]

    # Read files
    submission = pd.read_csv(submission_file)
    truth = pd.read_csv("data/test_labels.csv")  # shared labels

    must_have_columns(submission, ["id", "target"], "submission")
    must_have_columns(truth, ["id", "target"], "truth")

    # align by id
    merged = truth.merge(submission, on="id", suffixes=("_true", "_pred"))

    score = f1_score(merged["target_true"], merged["target_pred"], average="macro")
    print(f"Submission Macro-F1 Score: {score:.4f}")
