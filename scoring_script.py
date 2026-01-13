import sys
import pandas as pd
from sklearn.metrics import f1_score

# Usage: python scoring_script.py data/level_a/submissions/file.csv
submission_file = sys.argv[1]

submission = pd.read_csv(submission_file)

# ground truth MUST be a labels file, not test.csv
truth = pd.read_csv("data/level_a/test_labels.csv")

score = f1_score(truth["label"], submission["label"], average="macro")
print(f"Submission F1 Score: {score:.4f}")
