import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load training data
train = pd.read_csv('/Users/huden/Desktop/Study/minichallange/data/level_a/train.csv')

X = train.drop(['target'], axis=1)
y = train['target']

# Drop ID column from features
X = X.drop(['id'], axis=1)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a baseline model
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
score = f1_score(y_val, y_pred, average='macro')
print(f'Validation F1 Score: {score:.4f}')

# Load test data
test = pd.read_csv('/Users/huden/Desktop/Study/minichallange/data/level_a/test.csv')
test_ids = test['id']
X_test = test.drop(['id'], axis=1)

# Make predictions on test set
test_preds = clf.predict(X_test)

# Save submission
submission = pd.DataFrame({
    'id': test_ids,
    'target': test_preds
})

submission.to_csv(
    '/Users/huden/Desktop/Study/minichallange/submissions/sample_submission.csv',
    index=False
)

print('Saved: submissions/sample_submission.csv')
