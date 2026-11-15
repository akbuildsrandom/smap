# SmartMaintain - Machine Failure Prediction Model Training
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Dataset
df = pd.read_csv('data/ai4i2020.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Feature Engineering
# Feature 1: Temperature difference between process and ambient air (Heat Dissipation)
df['Temperature difference [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
# Feature 2: Mechanical Power in Watts = Torque * angular velocity (2 * pi * rpm / 60)
df['Mechanical power [W]'] = df['Torque [Nm]'] * (2 * np.pi * df['Rotational speed [rpm]'] / 60.0)
                                            
# 3. Separate Features (X) and Target (y)
# Exclude identifiers (UDI, Product ID) and failure modes (TWF, HDF, PWF, OSF, RNF) to prevent leakage
drop_columns = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
X = df.drop(columns=drop_columns)
y = df['Machine failure']

categorical_cols = ['Type']
numerical_cols = [
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
    'Torque [Nm]', 'Tool wear [min]', 'Temperature difference [K]', 'Mechanical power [W]'
]

# 4. Preprocessing: One-Hot Encoding for 'Type' and Standard Scaling for numeric features
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# 5. Stratified Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Train Only 2 Models using Scikit-Learn Pipelines
# Model 1: Logistic Regression
lr_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)

# Model 2: Random Forest Classifier
rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# 7. Evaluate Both Models
lr_preds = lr_pipeline.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"\n--- Model 1: Logistic Regression ---")
print(f"Accuracy:  {lr_acc:.4f} | Recall: {recall_score(y_test, lr_preds):.4f} | F1: {f1_score(y_test, lr_preds):.4f}")

rf_preds = rf_pipeline.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"\n--- Model 2: Random Forest Classifier ---")
print(f"Accuracy:  {rf_acc:.4f} | Recall: {recall_score(y_test, rf_preds):.4f} | F1: {f1_score(y_test, rf_preds):.4f}")

# 8. Compare & Select the More Accurate Model
if rf_acc >= lr_acc:
    best_pipeline, best_name, best_acc = rf_pipeline, "Random Forest Classifier", rf_acc
else:
    best_pipeline, best_name, best_acc = lr_pipeline, "Logistic Regression", lr_acc

print(f"\n[SELECTION] '{best_name}' selected with higher accuracy: {best_acc * 100:.2f}%")

# 9. Save Selected Model Pipeline
os.makedirs('models', exist_ok=True)
joblib.dump(best_pipeline, 'models/machine_failure_model.pkl')
print(f"[SUCCESS] Saved '{best_name}' pipeline to 'models/machine_failure_model.pkl'")
