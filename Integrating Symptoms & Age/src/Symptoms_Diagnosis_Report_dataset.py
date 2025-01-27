import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

report_templates = {
    "Bacterial Pneumonia": [
        "Patient diagnosed with bacterial pneumonia based on symptoms and confirmed by chest X-ray.",
        "Bacterial pneumonia detected; patient started on antibiotic therapy.",
        "Diagnosis confirmed through chest imaging and sputum culture for bacterial pneumonia.",
        "Patient exhibits typical signs of bacterial pneumonia; intravenous antibiotics initiated.",
        "Bacterial pneumonia confirmed via X-ray and clinical signs; patient monitored closely."
    ],
    "Viral Pneumonia": [
        "Patient exhibits signs of viral pneumonia, further analysis of symptoms indicates viral origin.",
        "Diagnosis indicates viral pneumonia; supportive care prescribed.",
        "Viral pneumonia suspected; PCR test for viruses pending.",
        "Patient presents with viral pneumonia symptoms; hydration and rest recommended.",
        "Viral pneumonia confirmed by viral load test; symptomatic treatment given."
    ],
    "Aspiration Pneumonia": [
        "Symptoms indicate aspiration pneumonia, typically caused by inhalation of foreign materials.",
        "Aspiration pneumonia confirmed; patient started on broad-spectrum antibiotics.",
        "Diagnosis of aspiration pneumonia; sputum culture pending for pathogen identification.",
        "Aspiration pneumonia suspected; patient monitored for respiratory complications.",
        "Patient exhibits signs of aspiration pneumonia; antibiotics targeting anaerobes initiated."
    ],
    "Staphylococcal Pneumonia": [
        "Confirmed staphylococcal pneumonia, requiring aggressive antibiotic therapy.",
        "Staphylococcal pneumonia diagnosed; vancomycin initiated for treatment.",
        "Diagnosis of staphylococcal pneumonia; patient managed with antibiotics.",
        "Staphylococcal pneumonia detected in patient; respiratory isolation implemented.",
        "Patient diagnosed with staphylococcal pneumonia; aggressive antibiotic therapy required."
    ],
    "Klebsiella Pneumonia": [
        "Patient diagnosed with klebsiella pneumonia; associated sepsis risk noted.",
        "Klebsiella pneumonia diagnosed; patient started on broad-spectrum antibiotics.",
        "Diagnosis of klebsiella pneumonia confirmed by sputum culture; managed with appropriate antibiotics.",
        "Klebsiella pneumonia confirmed; critical care support recommended due to sepsis risk.",
        "Klebsiella pneumonia detected; patient’s condition monitored for respiratory failure."
    ],
    "Eosinophilic Pneumonia": [
        "Eosinophilic pneumonia diagnosed; corticosteroid therapy initiated.",
        "Diagnosis of eosinophilic pneumonia confirmed; patient started on steroids.",
        "Eosinophilic pneumonia detected; patient monitored for response to corticosteroids.",
        "Eosinophilic pneumonia identified; patient started on anti-inflammatory therapy.",
        "Eosinophilic pneumonia diagnosed; careful monitoring of lung function recommended."
    ],
    "Lipid Pneumonia": [
        "Lipid pneumonia identified; often associated with inhalation of oils or fats.",
        "Diagnosis of lipid pneumonia confirmed; patient managed with supportive therapy.",
        "Lipid pneumonia suspected; history of inhalation of oily substances confirmed.",
        "Lipid pneumonia diagnosed; treatment focuses on avoiding further exposure to oils.",
        "Lipid pneumonia detected; patient's respiratory symptoms managed conservatively."
    ],
    "Interstitial Pneumonia": [
        "Interstitial pneumonia confirmed; fibrosis risk noted in long-term prognosis.",
        "Diagnosis of interstitial pneumonia; corticosteroid treatment initiated.",
        "Interstitial pneumonia detected on imaging; patient started on immunosuppressants.",
        "Interstitial pneumonia identified; management focuses on symptom control and lung preservation.",
        "Interstitial pneumonia diagnosed; lung biopsy confirmed the diagnosis."
    ],
    "Pseudomonas Pneumonia": [
        "Pseudomonas pneumonia identified, requiring antipseudomonal therapy.",
        "Diagnosis of pseudomonas pneumonia confirmed; patient started on piperacillin-tazobactam.",
        "Pseudomonas pneumonia detected; patient on ventilator support with antibiotics administered.",
        "Pseudomonas pneumonia diagnosed; respiratory culture showed growth of Pseudomonas aeruginosa.",
        "Pseudomonas pneumonia identified; management includes IV antibiotics and ventilation care."
    ],
    "Fungal Pneumonia": [
        "Fungal pneumonia diagnosed; antifungal treatment initiated.",
        "Diagnosis of fungal pneumonia confirmed by fungal culture; treatment with fluconazole prescribed.",
        "Fungal pneumonia detected; patient started on antifungal therapy and monitored closely.",
        "Fungal pneumonia identified; amphotericin B prescribed for treatment.",
        "Fungal pneumonia diagnosed; patient treated with targeted antifungal medication."
    ],
    "Postoperative Pneumonia": [
        "Postoperative pneumonia diagnosed in patient following surgery; treated with antibiotics.",
        "Diagnosis of postoperative pneumonia confirmed after signs of infection and chest X-ray findings.",
        "Patient developed pneumonia after surgery; started on empirical antibiotic therapy.",
        "Postoperative pneumonia identified; the patient is receiving IV antibiotics and respiratory care.",
        "Postoperative pneumonia diagnosed; patient is being monitored for complications and recovery."
    ],
    "Multilobar Pneumonia": [
        "Multilobar pneumonia diagnosed based on imaging findings of consolidation in multiple lung lobes.",
        "Patient diagnosed with multilobar pneumonia; broad-spectrum antibiotics initiated.",
        "Multilobar pneumonia confirmed through chest imaging; patient started on IV antibiotics.",
        "Diagnosis of multilobar pneumonia established; the patient is under intensive care and monitored closely.",
        "Multilobar pneumonia diagnosed; treatment started with targeted antibiotics and supportive care."
    ],
    "Neonatal Pneumonia": [
        "Neonatal pneumonia diagnosed in a newborn with respiratory distress and chest X-ray findings.",
        "Diagnosis of neonatal pneumonia confirmed; the patient is receiving antibiotics and respiratory support.",
        "Neonatal pneumonia diagnosed; the infant is under close monitoring in the neonatal intensive care unit.",
        "Neonatal pneumonia identified; antibiotic therapy and supportive care initiated.",
        "Diagnosis of neonatal pneumonia made; the patient is being treated with antibiotics and oxygen."
    ],
    "Aspiration of Gastric Contents Pneumonia": [
        "Aspiration pneumonia from gastric contents diagnosed based on clinical signs and chest X-ray.",
        "Diagnosis of aspiration pneumonia confirmed following aspiration event and imaging.",
        "Aspiration pneumonia from gastric contents identified; patient started on antibiotics and respiratory support.",
        "Aspiration pneumonia diagnosed; patient is receiving broad-spectrum antibiotics and airway management.",
        "Aspiration pneumonia caused by gastric contents; the patient is under observation with supportive care."
    ],
}

# this is every diagnosis, having its special set of sympyoms.
diagnosis_to_symptoms = {
    "Bacterial Pneumonia": ["fever", "cough", "shortness of breath", "bluish lips or face", "chest pain", "sputum production"],
    "Viral Pneumonia": ["fatigue", "dry cough", "joint pain", "sore throat", "fever", "headache"],
    "Aspiration Pneumonia": ["difficulty swallowing", "chest pain", "cough with foul-smelling sputum"],
    "Staphylococcal Pneumonia": ["high fever", "chills", "shortness of breath", "chest pain"],
    "Klebsiella Pneumonia": ["fever", "cough with bloody sputum", "chest pain", "confusion"],
    "Eosinophilic Pneumonia": ["wheezing", "cough", "fever", "shortness of breath", "fatigue"],
    "Lipid Pneumonia": ["chronic cough", "shortness of breath", "fever", "chest tightness"],
    "Interstitial Pneumonia": ["persistent dry cough", "fatigue", "shortness of breath", "chest tightness"],
    "Pseudomonas Pneumonia": ["fever", "cough with green sputum", "shortness of breath", "chest pain"],
    "Fungal Pneumonia": ["cough", "weight loss", "fever", "night sweats", "shortness of breath"],
    "Postoperative Pneumonia": ["fever", "cough", "chest pain", "shortness of breath", "fatigue"],
    "Multilobar Pneumonia": ["severe fever", "chest pain", "shortness of breath", "productive cough"],
    "Neonatal Pneumonia": ["rapid breathing", "grunting sounds", "fever", "poor feeding", "cyanosis"],
    "Aspiration of Gastric Contents Pneumonia": ["cough", "chest pain", "shortness of breath", "difficulty swallowing"],
}

# Define function for symptom augmentation (randomly add/remove symptoms or change them slightly)
def augment_symptoms(symptoms, diagnosis):
    augmented_symptoms = symptoms[:]
    available_symptoms = diagnosis_to_symptoms.get(diagnosis, [])
    
    # Randomly add or remove symptoms
    if random.random() < 0.5 and available_symptoms:
        symptom_to_add = random.choice(available_symptoms)
        if symptom_to_add not in augmented_symptoms:
            augmented_symptoms.append(symptom_to_add)

    if random.random() < 0.5 and augmented_symptoms:
        symptom_to_remove = random.choice(augmented_symptoms)
        augmented_symptoms.remove(symptom_to_remove)

    return augmented_symptoms

# Define function to augment reports (by slightly modifying them)
def augment_report(report):
    # Simple augmentations to modify the report slightly
    if "immediate treatment required" in report.lower():
        return report.replace("Immediate antibiotic treatment required.", "Immediate medical attention advised.")
    elif "rest and fluids recommended" in report.lower():
        return report.replace("Rest and fluids recommended.", "Patient should rest and hydrate well.")
    else:
        return report

# Generate random augmented data
samples = 1000

random_data = [
    {
        "Symptoms": ", ".join(augment_symptoms(
            random.sample(diagnosis_to_symptoms.get(diagnosis, ["unknown symptoms"]),
                          k=random.randint(1, len(diagnosis_to_symptoms[diagnosis]))),
            diagnosis
        )),
        "Diagnosis": diagnosis,
        "Report": augment_report(random.choice(report_templates.get(diagnosis, ["No report available"])))
    }
    for _ in range(samples)
    for diagnosis in [random.choice(list(diagnosis_to_symptoms.keys()))]
    for _ in range(samples)
    for diagnosis in [random.choice(list(diagnosis_to_symptoms.keys()))]
]

# Convert to DataFrame
df = pd.DataFrame(random_data)

# Split into training (80%) and testing (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Balance the training dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(
    train_df[["Symptoms", "Report"]], train_df["Diagnosis"]
)

# Combine resampled data into a DataFrame
train_df_resampled = pd.DataFrame({
    "Symptoms": X_train_resampled["Symptoms"],
    "Report": X_train_resampled["Report"],
    "Diagnosis": y_train_resampled
})

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
        return " ".join(tokens)
    else:
        return ""

train_df_resampled['Symptoms'] = train_df_resampled['Symptoms'].apply(preprocess_text)
test_df['Symptoms'] = test_df['Symptoms'].apply(preprocess_text)
train_df_resampled['Report'] = train_df_resampled['Report'].apply(preprocess_text)
test_df['Report'] = test_df['Report'].apply(preprocess_text)

vectorizer_symptoms = TfidfVectorizer()
X_train_symptoms = vectorizer_symptoms.fit_transform(train_df_resampled['Symptoms'])
X_test_symptoms = vectorizer_symptoms.transform(test_df['Symptoms'])

vectorizer_reports = TfidfVectorizer()
X_train_reports = vectorizer_reports.fit_transform(train_df_resampled['Report'])
X_test_reports = vectorizer_reports.transform(test_df['Report'])

X_train = hstack((X_train_symptoms, X_train_reports))
X_test = hstack((X_test_symptoms, X_test_reports))
y_train = train_df_resampled['Diagnosis']
y_test = test_df['Diagnosis']

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Create directories if they don't exist
os.makedirs("Integrating Symptoms & Age/synthetic data/dataset", exist_ok=True)

# Save datasets to CSV
train_df_resampled.to_csv("Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_training.csv", index=False)
test_df.to_csv("Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_testing.csv", index=False)

print("Dataset with augmented symptoms and reports saved.")
print(df.head())