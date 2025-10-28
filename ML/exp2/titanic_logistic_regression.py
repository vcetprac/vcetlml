import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set CSV file path
file_path = 'titanic_custom.csv'

# Check if file exists
if not os.path.isfile(file_path):
    print(f"Error: File '{file_path}' not found. Please place it in the current directory.")
    exit(1)

# Load dataset
df = pd.read_csv(file_path)

# Fill missing values (if any)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Map Sex (already 0/1 in this custom dataset)

# One-hot encode 'Embarked'
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True)

# Create features list dynamically including existing Embarked columns
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'] + list(embarked_dummies.columns)

X = df[features]
y = df['Survived']

# Scale Age and Fare
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression model training
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0,1]))

# Feature coefficients
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("\nFeature Coefficients:")
print(coef_df)

# Interactive user input for survival prediction
print("\nEnter passenger details to predict survival:")

def get_binary_input(prompt):
    while True:
        val = input(f"{prompt} (0 or 1): ").strip()
        if val in ['0', '1']:
            return int(val)
        print("Invalid input. Enter 0 or 1.")

def get_integer_input(prompt):
    while True:
        val = input(f"{prompt} (integer): ").strip()
        if val.isdigit():
            return int(val)
        print("Invalid input. Enter an integer.")

def get_float_input(prompt):
    while True:
        try:
            return float(input(f"{prompt} (number): ").strip())
        except ValueError:
            print("Invalid input. Enter a number.")

input_data = {}
for feature in features:
    if feature in ['Sex'] + list(embarked_dummies.columns):
        input_data[feature] = get_binary_input(feature)
    elif feature in ['Pclass', 'SibSp', 'Parch']:
        input_data[feature] = get_integer_input(feature)
    else:  # Age and Fare
        input_data[feature] = get_float_input(feature)

input_df = pd.DataFrame([input_data])
input_df[['Age', 'Fare']] = scaler.transform(input_df[['Age', 'Fare']])

survival_pred = model.predict(input_df)[0]
survival_prob = model.predict_proba(input_df)[0][survival_pred]

print("\nPrediction:")
if survival_pred == 1:
    print(f"Survived with probability {survival_prob:.2%}")
else:
    print(f"Did NOT survive with probability {survival_prob:.2%}")
