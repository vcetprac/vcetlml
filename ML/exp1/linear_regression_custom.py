import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
file_path = "housing_custom.csv"  # your dataset file path
df = pd.read_csv(file_path)
print(f"✅ Dataset Loaded | Shape: {df.shape}")
print(df)

# 2. Detect numeric columns for regression
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found for regression.")

# 3. Auto-select target (highest variance)
target_col = df[numeric_cols].var().idxmax()
print(f"✅ Detected target column: {target_col}")

# 4. Exploratory Data Analysis (EDA)
corr_matrix = df.corr()
print("\nCorrelation with target column:")
print(corr_matrix[target_col].sort_values(ascending=False))

plt.figure(figsize=(8,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title(f'Correlation Matrix (Target: {target_col})')
plt.show()

# 5. Feature Selection (top 5 correlated with target)
corr_with_target = corr_matrix[target_col].drop(target_col).abs()
selected_features = corr_with_target.sort_values(ascending=False).head(5).index.tolist()
print(f"✅ Selected features: {selected_features}")

X = df[selected_features]
y = df[target_col]

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# 8. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n--- PERFORMANCE METRICS ---")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2) Score: {r2:.3f}")

# Plot Actual vs Predicted values
plt.figure(figsize=(6,4))
sns.regplot(x=y_test, y=y_pred, line_kws={'color':'red'})
plt.xlabel(f'Actual {target_col}')
plt.ylabel(f'Predicted {target_col}')
plt.title('Actual vs Predicted (Regression Line)')
plt.show()

# 9. Predict on a custom input (first row as example)
sample_input = X.iloc[[0]]
predicted = lr.predict(sample_input)[0]
print(f"\nPredicted {target_col} for first entry: ${predicted:.2f}")
