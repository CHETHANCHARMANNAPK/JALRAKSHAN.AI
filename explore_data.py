import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("archive/smart_water_meter.csv")

df['result_time'] = pd.to_datetime(df['result_time'], errors='coerce')

plt.figure(figsize=(10,4))
plt.plot(df['result_time'], df['v1'])
plt.title("Smart Water Meter Consumption")
plt.xlabel("Time")
plt.ylabel("Water Usage (v1)")
plt.show()
df['leak'] = 0

leak_start = int(len(df) * 0.4)
leak_end = int(len(df) * 0.55)

df.loc[leak_start:leak_end, 'v1'] += 20
df.loc[leak_start:leak_end, 'leak'] = 1
