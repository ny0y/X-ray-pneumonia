import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Two lists that define what pneumonia symptoms are
PNEUMONIA_SYMPTOMS = ["fever", "cough", "shortness of breath", "chills", "fatigue", "chest pain", "high temperature", "tiredness", "wheezing"]
OTHER_SYMPTOMS = ["headache", "nausea", "sore throat", "dizziness", "runny nose"]

MINIMUM_AGE = 18
MAXIMUM_AGE = 90

# Combines both lists for the feature vector
ALL_SYMPTOMS = PNEUMONIA_SYMPTOMS + OTHER_SYMPTOMS

# Number of samples to generate
NUM_SAMPLES = 1000000

# Function to generate dataset
def generate_pneumonia_data(num_samples):
    data = []
    for _ in range(num_samples):
        # Assign a random age from 18 to 90
        age = random.randint(MINIMUM_AGE, MAXIMUM_AGE)
        # Choose if a sample has pneumonia or not at random
        if age > 65: # If patient is old
            # Assign a random number between 0 and 1
            # With a 30% of it being 0
            # And 70% chance of it being 1
            has_pneumonia = random.choices([0, 1], weights=(0.3, 0.7))
        else: # If patient is not old
            # Same but flipped chances
            has_pneumonia = random.choices([0, 1], weights=(0.7, 0.3))
        
        if has_pneumonia: # If it has pneumonia
            # Generate sample from the pneumonia list of size 2 to 9
            symptom_sample = random.sample(PNEUMONIA_SYMPTOMS, random.randint(2, len(PNEUMONIA_SYMPTOMS)))
        else: # If it doesn't have pneumonia
            # Generate sample from the pneumonia list of size 2 to 5
            symptom_sample = random.sample(OTHER_SYMPTOMS, random.randint(2, len(OTHER_SYMPTOMS)))
        
        # Create a feature vector based on the symptom set
        # If the symptom is in the sample, 1; otherwise, 0
        feature_vector = [1 if i in symptom_sample else 0 for i in ALL_SYMPTOMS]
        
        # Append to data as a dictionary
        data.append({"symptoms": symptom_sample, "label": has_pneumonia, "age": age, "features": feature_vector})
    
    return pd.DataFrame(data) # Return it in the form of a pandas DataFrame

# Generate the dataset
df = generate_pneumonia_data(NUM_SAMPLES) # Calls the method and returns the data to df
print("Sample synthetic data:")
print(df.head()) # Prints the first 5 elements of df

# Extract feature vectors and labels
X = np.array([list(row["features"]) + [row["age"]] for _, row in df.iterrows()]) # Feature vectors + the age
y = df["label"].values # The labels

# Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Convert the splits back into DataFrames for saving as CSV
train_df = pd.DataFrame(X_train, columns=ALL_SYMPTOMS + ["age"])
train_df["label"] = y_train

val_df = pd.DataFrame(X_val, columns=ALL_SYMPTOMS + ["age"])
val_df["label"] = y_val

test_df = pd.DataFrame(X_test, columns=ALL_SYMPTOMS + ["age"])
test_df["label"] = y_test

# Save datasets as CSV files
train_df.to_csv("Integrating Symptoms & Age/synthetic data/dataset/synthetic_data_train.csv", index=False)
val_df.to_csv("Integrating Symptoms & Age/synthetic data/dataset/synthetic_data_val.csv", index=False)
test_df.to_csv("Integrating Symptoms & Age/synthetic data/dataset/synthetic_data_test.csv", index=False)

print("Synthetic data generation and preprocessing complete.")
