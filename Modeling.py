import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Data Collection
data = pd.read_csv('path/to/your/nsl-kdd-dataset.csv')

# Data Preprocessing
# Assuming the dataset has a column 'label' indicating normal (0) or anomaly (1)
X = data.drop(columns=['label'])
y = data['label']

# Encoding categorical features
categorical_columns = X.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Development
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Model Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Converting predictions: -1 (anomalies) -> 1, 1 (normal) -> 0
y_pred_train = [1 if pred == -1 else 0 for pred in y_pred_train]
y_pred_test = [1 if pred == -1 else 0 for pred in y_pred_test]

# Classification report
print("Classification Report (Train):")
print(classification_report(y_train, y_pred_train))
print("Classification Report (Test):")
print(classification_report(y_test, y_pred_test))

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_test)
print(f'ROC-AUC Score: {roc_auc}')

# Precision, Recall, F1 Score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Plotting confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
import joblib
joblib.dump(model, 'anomaly_detection_model.joblib')
