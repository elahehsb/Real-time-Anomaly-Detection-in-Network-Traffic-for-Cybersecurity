import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

model = joblib.load('anomaly_detection_model.joblib')
scaler = joblib.load('scaler.joblib')
le_dict = joblib.load('label_encoders.joblib')

def preprocess_packet(packet):
    # Extract features from the packet
    features = {
        'src_ip': packet[IP].src,
        'dst_ip': packet[IP].dst,
        'src_port': packet.sport if TCP in packet or UDP in packet else 0,
        'dst_port': packet.dport if TCP in packet or UDP in packet else 0,
        'protocol': packet[IP].proto,
        'len': len(packet)
    }
    
    df = pd.DataFrame([features])
    for column in df.columns:
        if column in le_dict:
            df[column] = le_dict[column].transform(df[column])
    df = scaler.transform(df)
    return df

def detect_anomaly(packet):
    processed_data = preprocess_packet(packet)
    prediction = model.predict(processed_data)
    result = 'Anomaly' if prediction[0] == -1 else 'Normal'
    return result

def monitor_traffic():
    def process_packet(packet):
        if IP in packet:
            result = detect_anomaly(packet)
            print(f"Packet: {packet.summary()} - Prediction: {result}")
    
    scapy.sniff(prn=process_packet, store=False)

if __name__ == '__main__':
    monitor_traffic()
