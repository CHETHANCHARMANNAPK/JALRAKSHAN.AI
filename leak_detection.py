import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load and prepare data
df = pd.read_csv("archive/smart_water_meter.csv")
df['result_time'] = pd.to_datetime(df['result_time'], errors='coerce')

# Simulate leak in a portion of data
df['leak'] = 0
leak_start = int(len(df) * 0.4)
leak_end = int(len(df) * 0.55)
df.loc[leak_start:leak_end, 'v1'] += 20
df.loc[leak_start:leak_end, 'leak'] = 1

# Run anomaly detection
X = df[['v1']]
model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = model.fit_predict(X)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Show results
print(f"Total records: {len(df)}")
print(f"Anomalies detected: {df['anomaly'].sum()}")
print(f"Actual leaks: {df['leak'].sum()}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df['result_time'], df['v1'], label='Water Usage (v1)', alpha=0.7)
anomalies = df[df['anomaly'] == 1]
plt.scatter(anomalies['result_time'], anomalies['v1'], color='red', label='Anomaly (Leak)', s=10)
plt.title("Leak Detection using Isolation Forest")
plt.xlabel("Time")
plt.ylabel("Water Usage (v1)")
plt.legend()
plt.tight_layout()
plt.show()