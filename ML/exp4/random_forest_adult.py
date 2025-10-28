import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset
df = pd.read_csv("adult_custom.csv")
print(f"✅ Dataset Loaded | Shape: {df.shape}")

# 2. Data Cleaning
df = df.replace('?', np.nan)
df.dropna(inplace=True)
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 3. Auto-detect target column (binary column)
target_col = None
for col in df.columns:
    if df[col].nunique() == 2:
        target_col = col
        break
if target_col is None:
    raise ValueError("No binary target column found automatically!")
print(f"✅ Target column detected: {target_col}")

# 4. Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("✅ Label Encoding Completed")

# 5. Split data
X = df.drop(target_col, axis=1)
y = df[target_col]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data Split Completed")

# 6. Baseline Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n--- BASELINE RANDOM FOREST PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)

print("\n✅ Best Parameters Found:", grid_search.best_params_)

# 8. Optimized Random Forest model
best_rf = grid_search.best_estimator_
y_pred_opt = best_rf.predict(X_test)

print("\n--- OPTIMIZED RANDOM FOREST PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_opt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_opt):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_opt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_opt))

# 9. Feature Importance
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:\n", importances.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(10), y=importances.head(10).index, palette="crest")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 10. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 11. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoders[target_col].classes_,
            yticklabels=label_encoders[target_col].classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Optimized RF)')
plt.show()

# 12. Summary Report
baseline_acc = accuracy_score(y_test, y_pred)
optimized_acc = accuracy_score(y_test, y_pred_opt)
improvement = (optimized_acc - baseline_acc) * 100

print("\n--- SUMMARY REPORT ---")
print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
print(f"Optimized Accuracy: {optimized_acc*100:.2f}%")
print(f"Accuracy Improvement: {improvement:.2f}%")
print("\n✅ Random Forest Experiment Completed Successfully!")
