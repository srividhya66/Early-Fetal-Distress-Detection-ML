import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('fetal_health.csv')

# Handle potential issues with column names
column_mapping = {
    'baseline value': 'baseline_value',
    'accelerations': 'accelerations',
    'fetal_movement': 'fetal_movement',
    'uterine_contractions': 'uterine_contractions',
    'light_decelerations': 'light_decelerations',
    'severe_decelerations': 'severe_decelerations',
    'prolongued_decelerations': 'prolonged_decelerations',
    'abnormal_short_term_variability': 'abnormal_short_term_variability',
    'mean_value_of_short_term_variability': 'mean_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability': 'percentage_abnormal_long_term_variability',
    'mean_value_of_long_term_variability': 'mean_long_term_variability',
    'histogram_width': 'histogram_width',
    'histogram_min': 'histogram_min',
    'histogram_max': 'histogram_max',
    'histogram_number_of_peaks': 'histogram_peaks',
    'histogram_number_of_zeroes': 'histogram_zeroes',
    'histogram_mode': 'histogram_mode',
    'histogram_mean': 'histogram_mean',
    'histogram_median': 'histogram_median',
    'histogram_variance': 'histogram_variance',
    'histogram_tendency': 'histogram_tendency',
    'fetal_health': 'fetal_health'
}

df = df.rename(columns=column_mapping)

print(f"Original data shape: {df.shape}")

# Select relevant features and target
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'].astype(int)

# Scale the features - save the scaler for use in predictions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.23, random_state=42, stratify=y)

# Increase the number of trees and adjust class weights to handle class imbalance
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on training set
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy:.4f}")

# Evaluate on test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Test the specific pathological case
pathological_case = [[134, 0.001, 0, 0.013, 0.008, 0, 0.003, 29, 6.3, 0, 0, 150, 50, 200, 6, 3, 71, 107, 106, 215, 0]]
pathological_scaled = scaler.transform(pathological_case)
pathological_pred = model.predict(pathological_scaled)
print(f"\nPrediction for pathological test case: {pathological_pred[0]} (Should be 3)")

# Save the model and scaler
joblib.dump(model, "fetal_model.pkl")
joblib.dump(scaler, "fetal_scaler.pkl")
print("\nModel and scaler saved as 'fetal_model.pkl' and 'fetal_scaler.pkl'")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))