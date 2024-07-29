from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

app = Flask(__name__)
model = joblib.load('anomaly_detection_model.joblib')
scaler = joblib.load('scaler.joblib')
le_dict = joblib.load('label_encoders.joblib')

def preprocess_data(data):
    df = pd.DataFrame(data, index=[0])
    for column in df.columns:
        if column in le_dict:
            df[column] = le_dict[column].transform(df[column])
    df = scaler.transform(df)
    return df

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    processed_data = preprocess_data(data)
    prediction = model.predict(processed_data)
    result = 'Anomaly' if prediction[0] == -1 else 'Normal'
    
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
