import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset
df = pd.read_csv("adult_custom.csv")
print(f"\n✅ Dataset Loaded | Shape: {df.shape}")

# 2. Data Cleaning
df = df.replace('?', np.nan)
df.dropna(inplace=True)
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 3. Auto-detect target (assumes binary with 2 unique values)
target_col = None
for col in df.columns[::-1]:
    if df[col].nunique() == 2:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]
print(f"🎯 Target column automatically detected as: '{target_col}'")

# 4. Encode Categorical Columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("✅ Label Encoding Completed")

# 5. Split Data
X = df.drop(target_col, axis=1)
y = df[target_col]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data Split Done")

# 6. Base Model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- BASELINE MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Confusion Matrix (Baseline)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Baseline Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Hyperparameter Tuning
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)
print("\nBest Parameters Found:", grid_search.best_params_)

# 9. Optimized Model
best_clf = grid_search.best_estimator_
y_pred_opt = best_clf.predict(X_test)

print("\n--- OPTIMIZED MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_opt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_opt):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_opt))

# 10. Confusion Matrix (Optimized)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Optimized Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 11. Feature Importance
importances = pd.Series(best_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 Important Features:\n", importances.head())

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.head(10).values, y=importances.head(10).index)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# 12. Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(
    best_clf,
    feature_names=list(X.columns),
    class_names=[str(cls) for cls in np.unique(y)],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree Visualization (Optimized Model)")
plt.tight_layout()
plt.show()

