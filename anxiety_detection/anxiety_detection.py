import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train_path = "panic_disorder_dataset_training.csv"
test_path = "panic_disorder_dataset_testing.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display first few rows to inspect data
print(train_df.head())

# Identify categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])  # Apply same encoding to test data
    label_encoders[col] = le  # Save encoder for future use

# Separate features and target variable
X_train = train_df.drop(columns=["Panic Disorder Diagnosis"])  # Replace "Target" with the actual target column name
y_train = train_df["Panic Disorder Diagnosis"]
X_test = test_df.drop(columns=["Panic Disorder Diagnosis"])
y_test = test_df["Panic Disorder Diagnosis"]

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
