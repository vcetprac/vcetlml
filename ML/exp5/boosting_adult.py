import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

# 1. Load dataset
file_path = "adult_custom.csv"
df = pd.read_csv(file_path)
print(f"✅ Dataset Loaded | Shape: {df.shape}")

# 2. Automatically detect target column (binary string or last column)
target_col = None
for col in df.columns:
    if df[col].nunique() == 2 and df[col].dtype == object:
        target_col = col
        break
else:
    target_col = df.columns[-1]

print(f"✅ Target column detected: {target_col}")

# 3. Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode target if categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data Split Done")

# 5. Apply boosting algorithm
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- BOOSTING MODEL PERFORMANCE ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# 7. Classification report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, zero_division=0))
