# AI-HPC-Predictive-Maintenance
This project leverages Machine Learning (Random Forest) to predict failures in HPC systems by analyzing real-time system logs. The goal is to minimize downtime, optimize resource usage, and improve system reliability.
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Generate Synthetic System Log Data
def generate_synthetic_data(n=10000):
    np.random.seed(42)
    random.seed(42)
    
    cpu_usage = np.random.randint(10, 100, n)
    memory_usage = np.random.randint(10, 100, n)
    disk_io = np.random.randint(50, 500, n)
    temperature = np.random.randint(30, 90, n)
    power_consumption = np.random.randint(50, 300, n)
    
    # Generating random failures (1 = failure, 0 = no failure)
    failure = [1 if (cpu > 85 and temp > 75) or (power > 250) else 0 
               for cpu, temp, power in zip(cpu_usage, temperature, power_consumption)]
    
    data = pd.DataFrame({
        'CPU_Usage': cpu_usage,
        'Memory_Usage': memory_usage,
        'Disk_IO': disk_io,
        'Temperature': temperature,
        'Power_Consumption': power_consumption,
        'Failure': failure
    })
    return data

# Generate data
data = generate_synthetic_data()

# Step 2: Train Machine Learning Model
X = data[['CPU_Usage', 'Memory_Usage', 'Disk_IO', 'Temperature', 'Power_Consumption']]
y = data['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Step 3: Real-time Failure Prediction
def predict_failure(cpu, memory, disk_io, temp, power):
    input_data = np.array([[cpu, memory, disk_io, temp, power]])
    prediction = model.predict(input_data)[0]
    return 'Failure Detected!' if prediction == 1 else 'System Healthy'

# Simulating real-time monitoring
sample_input = [88, 50, 300, 80, 270]  # Example values
print("Real-time Prediction:", predict_failure(*sample_input))
