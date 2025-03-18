import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE  # For handling class imbalance

# Load dataset
file_path = "stroke_risk_dataset.csv"
data = pd.read_csv(file_path)

# Handle missing values
data.fillna(data.median(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop("At Risk (Binary)", axis=1)  # Ensure target column is not in features
y = data["At Risk (Binary)"]

# Check for data leakage
assert "At Risk (Binary)" not in X.columns, "Target column is leaking into features!"

# Inspect the dataset
print("Target variable distribution:")
print(y.value_counts())

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature selection to remove less important features
selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42))
selector.fit(X, y)
X = selector.transform(X)

# Check number of features after selection
print(f"Number of features after selection: {X.shape[1]}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train a Random Forest with balanced hyperparameters
model = RandomForestClassifier(
    n_estimators=100,          # Increase number of trees
    max_depth=None,             # Allow deeper trees
    min_samples_split=2,       # Reduce minimum samples for splits
    min_samples_leaf=1,        # Reduce minimum samples for leaves
    max_features="auto",        # Use all features for splits
    class_weight="balanced",   # Handle class imbalance
    random_state=42
)

# Use cross-validation to check for overfitting
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))