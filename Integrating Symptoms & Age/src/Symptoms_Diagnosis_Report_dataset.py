import pandas as pd
import random

# Provided lists
symptoms_list = [
    "chest pain", "fever", "cough", "fatigue", "shortness of breath",
    "difficulty breathing", "chills", "sore throat", "loss of appetite",
    "muscle aches", "headache", "confusion", "bluish lips or face",
    "rapid breathing", "wheezing", "sputum production", "stomach pain",
    "vomiting", "diarrhea", "night sweats", "joint pain", "neck stiffness",
    "nausea", "rash", "weight loss", "body aches", "malaise", "dry cough",
    "productive cough", "blood in sputum", "hoarseness", "rapid heartbeat",
    "shivering", "swollen lymph nodes", "yellowish skin", "congested nose",
    "ear pain", "eye redness", "decreased urination", "paleness",
    "irritability", "loss of smell", "loss of taste", "back pain",
    "shoulder pain", "dizziness", "unsteady gait", "difficulty swallowing",
    "general weakness", "chest tightness"
]

diagnoses_list = [
    "Bacterial Pneumonia", "Viral Pneumonia", "Mycoplasma Pneumonia", "Pneumonia NOS",
    "Aspiration Pneumonia", "Staphylococcal Pneumonia", "Klebsiella Pneumonia",
    "Pneumocystis Pneumonia", "Rickettsial Pneumonia", "Eosinophilic Pneumonia",
    "Ventilator-Associated Pneumonia", "Lipid Pneumonia", "Bronchopneumonia",
    "Interstitial Pneumonia", "Community-Acquired Pneumonia", "Hospital-Acquired Pneumonia",
    "Gram-Negative Bacterial Pneumonia", "Pseudomonas Pneumonia", "Fungal Pneumonia",
    "Legionella Pneumonia", "Tuberculosis-Related Pneumonia", "Chemical Pneumonia",
    "Postoperative Pneumonia", "Multilobar Pneumonia", "Atypical Pneumonia",
    "MERS-Associated Pneumonia", "H1N1 Pneumonia", "COVID-19 Pneumonia",
    "Pediatric Pneumonia", "Geriatric Pneumonia", "Immunocompromised Pneumonia",
    "Neonatal Pneumonia", "Chlamydial Pneumonia", "Respiratory Syncytial Virus Pneumonia",
    "Anthrax Pneumonia", "Severe Acute Respiratory Syndrome Pneumonia",
    "Influenza Pneumonia", "Zoonotic Pneumonia", "Plague-Related Pneumonia",
    "Vibrio Pneumonia", "Cystic Fibrosis-Associated Pneumonia", "Bronchiolitis Obliterans Pneumonia",
    "Aspiration of Gastric Contents Pneumonia", "Pneumonia Caused by Lipoid Aspiration",
    "Pneumonia from Overdose", "Sepsis-Related Pneumonia", "Lung Abscess Pneumonia",
    "Opportunistic Fungal Pneumonia", "Chronic Pneumonia", "Recurrent Pneumonia"
]

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
    "Mycoplasma Pneumonia": [
        "Diagnosis indicates mycoplasma pneumonia, treatment with antibiotics recommended.",
        "Patient diagnosed with mycoplasma pneumonia based on clinical presentation and chest imaging.",
        "Mycoplasma pneumonia identified; doxycycline prescribed for treatment.",
        "Symptoms and chest X-ray findings are consistent with mycoplasma pneumonia.",
        "Patient diagnosed with mycoplasma pneumonia; observed for response to antibiotics."
    ],
    "Pneumonia NOS": [
        "Diagnosis of pneumonia NOS, further investigations required to identify pathogen.",
        "Pneumonia NOS diagnosed; further microbiological tests recommended.",
        "Unidentified pathogen causing pneumonia; empirical antibiotics prescribed.",
        "Clinical features suggest pneumonia NOS; pathogen identification ongoing.",
        "Pneumonia NOS confirmed; patient to be closely monitored for changes in condition."
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
    "Pneumocystis Pneumonia": [
        "Pneumocystis pneumonia detected, indicative of immune system suppression.",
        "Diagnosis of pneumocystis pneumonia confirmed; patient started on anti-fungal treatment.",
        "Pneumocystis pneumonia diagnosed; bronchoscopy and biopsy performed for confirmation.",
        "Pneumocystis pneumonia identified; patient treated with high-dose cotrimoxazole.",
        "Pneumocystis pneumonia suspected; treatment with antifungal agents recommended."
    ],
    "Rickettsial Pneumonia": [
        "Diagnosis consistent with rickettsial pneumonia; zoonotic exposure suspected.",
        "Rickettsial pneumonia diagnosed; doxycycline prescribed for treatment.",
        "Patient diagnosed with rickettsial pneumonia; tick exposure history confirmed.",
        "Rickettsial pneumonia confirmed by serology; patient treated with appropriate antibiotics.",
        "Rickettsial pneumonia detected; supportive care along with antimicrobial therapy recommended."
    ],
    "Eosinophilic Pneumonia": [
        "Eosinophilic pneumonia diagnosed; corticosteroid therapy initiated.",
        "Diagnosis of eosinophilic pneumonia confirmed; patient started on steroids.",
        "Eosinophilic pneumonia detected; patient monitored for response to corticosteroids.",
        "Eosinophilic pneumonia identified; patient started on anti-inflammatory therapy.",
        "Eosinophilic pneumonia diagnosed; careful monitoring of lung function recommended."
    ],
    "Ventilator-Associated Pneumonia": [
        "Patient has ventilator-associated pneumonia; ventilator settings reviewed.",
        "Diagnosis of ventilator-associated pneumonia confirmed; antibiotics adjusted.",
        "Ventilator-associated pneumonia diagnosed; microbiological analysis of sputum pending.",
        "Patient with ventilator-associated pneumonia; infection control measures implemented.",
        "Ventilator-associated pneumonia identified; supportive care and broad-spectrum antibiotics administered."
    ],
    "Lipid Pneumonia": [
        "Lipid pneumonia identified; often associated with inhalation of oils or fats.",
        "Diagnosis of lipid pneumonia confirmed; patient managed with supportive therapy.",
        "Lipid pneumonia suspected; history of inhalation of oily substances confirmed.",
        "Lipid pneumonia diagnosed; treatment focuses on avoiding further exposure to oils.",
        "Lipid pneumonia detected; patient's respiratory symptoms managed conservatively."
    ],
    "Bronchopneumonia": [
        "Diffuse bronchopneumonia visible on imaging; prescribed supportive care.",
        "Diagnosis of bronchopneumonia confirmed; broad-spectrum antibiotics initiated.",
        "Patient diagnosed with bronchopneumonia; observed for response to treatment.",
        "Bronchopneumonia detected on imaging; treated with antibiotics targeting pathogens.",
        "Bronchopneumonia identified; supportive care and oxygen therapy recommended."
    ],
    "Interstitial Pneumonia": [
        "Interstitial pneumonia confirmed; fibrosis risk noted in long-term prognosis.",
        "Diagnosis of interstitial pneumonia; corticosteroid treatment initiated.",
        "Interstitial pneumonia detected on imaging; patient started on immunosuppressants.",
        "Interstitial pneumonia identified; management focuses on symptom control and lung preservation.",
        "Interstitial pneumonia diagnosed; lung biopsy confirmed the diagnosis."
    ],
    "Community-Acquired Pneumonia": [
        "Diagnosis: community-acquired pneumonia; outpatient management recommended.",
        "Patient diagnosed with community-acquired pneumonia; oral antibiotics prescribed.",
        "Community-acquired pneumonia detected; patient observed for worsening symptoms.",
        "Patient exhibits signs of community-acquired pneumonia; managed with antibiotics and rest.",
        "Community-acquired pneumonia confirmed; patient treated as outpatient with antibiotic regimen."
    ],
    "Hospital-Acquired Pneumonia": [
        "Patient diagnosed with hospital-acquired pneumonia, requiring broad-spectrum antibiotics.",
        "Hospital-acquired pneumonia confirmed; patient started on IV antibiotics.",
        "Diagnosis of hospital-acquired pneumonia; treatment with extended-spectrum antibiotics initiated.",
        "Patient with hospital-acquired pneumonia; respiratory isolation and monitoring advised.",
        "Hospital-acquired pneumonia diagnosed; targeted antimicrobial therapy required."
    ],
    "Gram-Negative Bacterial Pneumonia": [
        "Gram-negative bacterial pneumonia detected, resistance profile assessed.",
        "Diagnosis of gram-negative pneumonia confirmed; patient started on appropriate antibiotics.",
        "Gram-negative bacterial pneumonia diagnosed; sensitivity testing performed for antibiotic selection.",
        "Patient with gram-negative bacterial pneumonia; ventilation support and antibiotics provided.",
        "Gram-negative pneumonia confirmed; culture results pending for further pathogen identification."
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
    "Legionella Pneumonia": [
        "Legionella pneumonia confirmed via culture and urine antigen test.",
        "Diagnosis of Legionella pneumonia confirmed; patient started on levofloxacin.",
        "Legionella pneumonia identified; clinical presentation and urine antigen test confirm diagnosis.",
        "Legionella pneumonia detected; appropriate antibiotics initiated immediately.",
        "Patient diagnosed with Legionella pneumonia; supportive care and antibiotics administered."
    ],
    "Tuberculosis-Related Pneumonia": [
        "Pneumonia secondary to tuberculosis diagnosed; long-term therapy planned.",
        "Diagnosis of tuberculosis-related pneumonia confirmed; patient started on antitubercular drugs.",
        "Tuberculosis-related pneumonia identified; patient placed on a combination therapy regimen.",
        "Pneumonia due to tuberculosis diagnosed; patient started on multi-drug regimen for TB.",
        "Tuberculosis-related pneumonia confirmed; patient managed in isolation with specialized treatment."
    ],
    "Chronic Obstructive Pulmonary Disease (COPD)": [
        "Chronic obstructive pulmonary disease exacerbation diagnosed; corticosteroids and bronchodilators initiated.",
        "COPD flare-up identified; patient managed with bronchodilators and respiratory therapy.",
        "Diagnosis of COPD exacerbation confirmed; oral corticosteroids prescribed for symptom relief.",
        "Patient diagnosed with COPD exacerbation; pulmonary rehabilitation and inhalers prescribed.",
        "COPD diagnosed; patient started on long-acting beta agonists and monitored for lung function."
    ],
    "Pulmonary Embolism": [
        "Pulmonary embolism suspected; CT pulmonary angiogram ordered for confirmation.",
        "Diagnosis of pulmonary embolism confirmed; patient started on anticoagulants.",
        "Pulmonary embolism detected via imaging; treatment with heparin initiated.",
        "Patient diagnosed with pulmonary embolism; thrombolytic therapy recommended.",
        "Pulmonary embolism confirmed; patient under close monitoring and anticoagulant therapy."
    ],
    "Chemical Pneumonia": [
        "Chemical pneumonia diagnosed based on exposure history and chest imaging findings.",
        "Patient diagnosed with chemical pneumonia following inhalation of toxic fumes; supportive treatment initiated.",
        "Diagnosis of chemical pneumonia confirmed; patient is receiving respiratory support and corticosteroids.",
        "Chemical pneumonia identified; patient started on bronchodilators and oxygen therapy.",
        "Chemical pneumonia diagnosed; patient is under close observation with respiratory management."
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
    "Atypical Pneumonia": [
        "Atypical pneumonia diagnosed based on clinical presentation and imaging results.",
        "Diagnosis of atypical pneumonia confirmed with PCR testing for Mycoplasma or Chlamydia species.",
        "Patient diagnosed with atypical pneumonia; treatment with azithromycin started.",
        "Atypical pneumonia identified; patient is receiving outpatient care and antibiotics.",
        "Atypical pneumonia diagnosed; supportive care and antibiotics initiated."
    ],
    "MERS-Associated Pneumonia": [
        "MERS-associated pneumonia diagnosed following exposure history and positive PCR test.",
        "Patient diagnosed with MERS-associated pneumonia; the patient is being isolated and treated with antivirals.",
        "Diagnosis of MERS-associated pneumonia confirmed; patient is receiving supportive care and antiviral therapy.",
        "MERS-associated pneumonia diagnosed; the patient is under strict isolation and respiratory support.",
        "Patient diagnosed with MERS-associated pneumonia; antiviral therapy and monitoring initiated."
    ],
    "H1N1 Pneumonia": [
        "H1N1 pneumonia diagnosed based on PCR results and chest X-ray findings.",
        "Diagnosis of H1N1 pneumonia confirmed; the patient is receiving antiviral treatment and respiratory support.",
        "H1N1 pneumonia identified in patient with typical flu symptoms and imaging findings.",
        "Patient diagnosed with H1N1 pneumonia; antiviral therapy initiated along with oxygen support.",
        "H1N1 pneumonia confirmed; supportive care and antiviral therapy administered."
    ],
    "COVID-19 Pneumonia": [
        "COVID-19 pneumonia diagnosed in patient with positive PCR test and characteristic lung infiltrates.",
        "Patient diagnosed with COVID-19 pneumonia; started on antiviral therapy and oxygen therapy.",
        "COVID-19 pneumonia confirmed; patient is receiving mechanical ventilation and antiviral treatment.",
        "Diagnosis of COVID-19 pneumonia made; patient is being treated with remdesivir and supportive care.",
        "COVID-19 pneumonia diagnosed; the patient is under isolation with ongoing respiratory support."
    ],
    "Pediatric Pneumonia": [
        "Pediatric pneumonia diagnosed in a young patient with fever, cough, and imaging findings.",
        "Diagnosis of pediatric pneumonia made based on clinical presentation and chest X-ray.",
        "Patient diagnosed with pediatric pneumonia; antibiotics and supportive care initiated.",
        "Pediatric pneumonia confirmed; the patient is recovering with outpatient antibiotics.",
        "Pediatric pneumonia diagnosed; patient is being monitored with supportive treatments."
    ],
    "Geriatric Pneumonia": [
        "Geriatric pneumonia diagnosed in elderly patient with fever and confusion; antibiotics started.",
        "Diagnosis of geriatric pneumonia confirmed in elderly patient; treated with IV antibiotics and supportive care.",
        "Geriatric pneumonia diagnosed; patient started on antibiotics and oxygen therapy.",
        "Patient diagnosed with geriatric pneumonia; the treatment plan includes antibiotics and respiratory management.",
        "Geriatric pneumonia confirmed; patient is under close monitoring with intravenous antibiotics."
    ],
    "Immunocompromised Pneumonia": [
        "Immunocompromised pneumonia diagnosed in patient with weakened immune system; treated with broad-spectrum antibiotics.",
        "Diagnosis of pneumonia in immunocompromised patient confirmed; antifungal and antibiotic therapy initiated.",
        "Immunocompromised pneumonia identified; patient started on prophylactic antibiotics and antifungals.",
        "Patient with immunocompromised pneumonia; the patient is being treated with empiric antibiotic therapy.",
        "Immunocompromised pneumonia diagnosed; ongoing monitoring and supportive care provided."
    ],
    "Neonatal Pneumonia": [
        "Neonatal pneumonia diagnosed in a newborn with respiratory distress and chest X-ray findings.",
        "Diagnosis of neonatal pneumonia confirmed; the patient is receiving antibiotics and respiratory support.",
        "Neonatal pneumonia diagnosed; the infant is under close monitoring in the neonatal intensive care unit.",
        "Neonatal pneumonia identified; antibiotic therapy and supportive care initiated.",
        "Diagnosis of neonatal pneumonia made; the patient is being treated with antibiotics and oxygen."
    ],
    "Chlamydial Pneumonia": [
        "Chlamydial pneumonia diagnosed based on positive PCR test and chest imaging.",
        "Diagnosis of chlamydial pneumonia confirmed; patient started on doxycycline or azithromycin.",
        "Chlamydial pneumonia diagnosed in patient with typical symptoms; antibiotics initiated.",
        "Patient diagnosed with chlamydial pneumonia; treatment with doxycycline started.",
        "Chlamydial pneumonia confirmed; patient is being treated with appropriate antibiotics."
    ],
    "Respiratory Syncytial Virus Pneumonia": [
        "RSV pneumonia diagnosed based on PCR results and clinical symptoms.",
        "Diagnosis of respiratory syncytial virus pneumonia confirmed; the patient is receiving supportive care.",
        "RSV pneumonia identified in a pediatric patient; oxygen therapy and hydration initiated.",
        "Respiratory syncytial virus pneumonia diagnosed; the patient is under respiratory support and observation.",
        "RSV pneumonia diagnosed; supportive treatment and monitoring started for respiratory distress."
    ],
    "Anthrax Pneumonia": [
        "Anthrax pneumonia diagnosed following exposure history and chest imaging findings.",
        "Diagnosis of anthrax pneumonia confirmed; patient started on appropriate antibiotic therapy.",
        "Patient diagnosed with anthrax pneumonia; receiving a combination of antibiotics and antitoxin treatment.",
        "Anthrax pneumonia identified; treatment initiated with ciprofloxacin and supportive care.",
        "Anthrax pneumonia confirmed; patient is under observation and receiving intensive care."
    ],
    "Severe Acute Respiratory Syndrome Pneumonia": [
        "SARS pneumonia diagnosed based on positive PCR test and chest X-ray showing infiltrates.",
        "Diagnosis of SARS pneumonia confirmed; the patient is receiving antiviral and supportive therapy.",
        "SARS pneumonia diagnosed; patient started on antivirals and respiratory support.",
        "Patient diagnosed with SARS pneumonia; the treatment includes antiviral agents and oxygen therapy.",
        "SARS pneumonia confirmed; patient is under isolation and intensive monitoring."
    ],
    "Influenza Pneumonia": [
        "Influenza pneumonia diagnosed based on PCR results and chest imaging.",
        "Diagnosis of influenza pneumonia confirmed; antiviral therapy started along with supportive care.",
        "Patient diagnosed with influenza pneumonia; supportive care and antivirals initiated.",
        "Influenza pneumonia diagnosed; the patient is receiving oxygen therapy and rest.",
        "Influenza pneumonia identified; patient is receiving antiviral treatment and hydration."
    ],
    "Zoonotic Pneumonia": [
        "Zoonotic pneumonia diagnosed following animal exposure history and chest imaging.",
        "Diagnosis of zoonotic pneumonia confirmed based on exposure history and laboratory tests.",
        "Patient diagnosed with zoonotic pneumonia; antibiotics and antivirals initiated.",
        "Zoonotic pneumonia identified; the patient is being treated with targeted antibiotics and antivirals.",
        "Zoonotic pneumonia diagnosed; the patient is under isolation with ongoing treatment."
    ],
    "Plague-Related Pneumonia": [
        "Plague-related pneumonia diagnosed based on exposure history and clinical presentation.",
        "Diagnosis of plague-related pneumonia confirmed; patient started on appropriate antibiotics.",
        "Plague-related pneumonia diagnosed; the patient is receiving intensive care and antibiotics.",
        "Plague-related pneumonia identified; antibiotic therapy and isolation initiated.",
        "Plague-related pneumonia confirmed; patient is under strict isolation and receiving supportive care."
    ],
    "Vibrio Pneumonia": [
        "Vibrio pneumonia diagnosed following exposure to contaminated water and chest imaging.",
        "Diagnosis of vibrio pneumonia confirmed; the patient is being treated with appropriate antibiotics.",
        "Vibrio pneumonia diagnosed; patient started on antibiotics and closely monitored.",
        "Vibrio pneumonia confirmed; patient is under intensive care and receiving intravenous antibiotics.",
        "Vibrio pneumonia diagnosed; the patient is responding to antibiotic therapy."
    ],
    "Cystic Fibrosis-Associated Pneumonia": [
        "Cystic fibrosis-associated pneumonia diagnosed in patient with chronic lung disease.",
        "Diagnosis of cystic fibrosis-associated pneumonia confirmed by sputum culture and chest imaging.",
        "Patient diagnosed with cystic fibrosis-associated pneumonia; antibiotics and airway clearance initiated.",
        "Cystic fibrosis-associated pneumonia identified; patient started on antibiotics and bronchodilators.",
        "Cystic fibrosis-associated pneumonia diagnosed; the patient is receiving intensive care and respiratory therapy."
    ],
    "Bronchiolitis Obliterans Pneumonia": [
        "Bronchiolitis obliterans pneumonia diagnosed based on history of lung injury and chest imaging.",
        "Diagnosis of bronchiolitis obliterans pneumonia confirmed through clinical and imaging studies.",
        "Patient diagnosed with bronchiolitis obliterans pneumonia; treatment with corticosteroids started.",
        "Bronchiolitis obliterans pneumonia diagnosed; patient is receiving immunosuppressive therapy.",
        "Bronchiolitis obliterans pneumonia identified; the patient is under monitoring with ongoing therapy."
    ],
    "Aspiration of Gastric Contents Pneumonia": [
        "Aspiration pneumonia from gastric contents diagnosed based on clinical signs and chest X-ray.",
        "Diagnosis of aspiration pneumonia confirmed following aspiration event and imaging.",
        "Aspiration pneumonia from gastric contents identified; patient started on antibiotics and respiratory support.",
        "Aspiration pneumonia diagnosed; patient is receiving broad-spectrum antibiotics and airway management.",
        "Aspiration pneumonia caused by gastric contents; the patient is under observation with supportive care."
    ],
    "Pneumonia Caused by Lipoid Aspiration": [
        "Pneumonia caused by lipoid aspiration diagnosed following history of aspiration and chest imaging.",
        "Diagnosis of lipoid aspiration pneumonia confirmed; patient is receiving corticosteroid therapy.",
        "Lipoid aspiration pneumonia diagnosed; patient started on corticosteroids and supportive care.",
        "Pneumonia caused by lipoid aspiration identified; treatment with corticosteroids initiated.",
        "Lipoid aspiration pneumonia diagnosed; the patient is recovering with respiratory therapy."
    ],
    "Pneumonia from Overdose": [
        "Pneumonia resulting from overdose diagnosed following clinical signs and imaging studies.",
        "Diagnosis of pneumonia from overdose confirmed; patient started on supportive care and respiratory management.",
        "Pneumonia from overdose diagnosed; the patient is under observation with treatment for underlying condition.",
        "Pneumonia from overdose identified; supportive treatment and detoxification initiated.",
        "Pneumonia caused by overdose diagnosed; the patient is receiving respiratory support."
    ],
    "Sepsis-Related Pneumonia": [
        "Sepsis-related pneumonia diagnosed in patient with systemic infection and chest imaging abnormalities.",
        "Diagnosis of sepsis-related pneumonia confirmed; the patient is under intensive care with antibiotics.",
        "Sepsis-related pneumonia diagnosed; the patient started on broad-spectrum antibiotics and IV fluids.",
        "Sepsis-related pneumonia identified; patient is receiving mechanical ventilation and supportive care.",
        "Sepsis-related pneumonia diagnosed; patient is monitored in the ICU with ongoing antibiotic therapy."
    ],
    "Lung Abscess Pneumonia": [
        "Lung abscess pneumonia diagnosed based on chest imaging findings of a cavity and consolidation.",
        "Diagnosis of lung abscess pneumonia confirmed; patient started on appropriate antibiotics and drainage.",
        "Lung abscess pneumonia diagnosed; patient undergoing antibiotic therapy and surgical drainage.",
        "Patient diagnosed with lung abscess pneumonia; IV antibiotics and chest drainage initiated.",
        "Lung abscess pneumonia diagnosed; patient is under observation and receiving intensive antibiotic therapy."
    ],
    "Opportunistic Fungal Pneumonia": [
        "Opportunistic fungal pneumonia diagnosed in immunocompromised patient based on sputum culture.",
        "Diagnosis of fungal pneumonia confirmed; the patient is receiving antifungal treatment and supportive care.",
        "Opportunistic fungal pneumonia diagnosed; patient started on antifungal therapy and respiratory support.",
        "Fungal pneumonia diagnosed in immunocompromised patient; antifungal therapy and monitoring initiated.",
        "Opportunistic fungal pneumonia identified; the patient is being treated with appropriate antifungal agents."
    ],
    "Chronic Pneumonia": [
        "Chronic pneumonia diagnosed based on long-standing symptoms and chest imaging findings.",
        "Diagnosis of chronic pneumonia confirmed; patient is receiving long-term antibiotics and respiratory care.",
        "Chronic pneumonia diagnosed; patient is under observation with ongoing management of symptoms.",
        "Patient diagnosed with chronic pneumonia; treatment includes antibiotics and bronchodilators.",
        "Chronic pneumonia diagnosed; the patient is being monitored with long-term respiratory therapy."
    ],
    "Recurrent Pneumonia": [
        "Recurrent pneumonia diagnosed in patient with multiple episodes of infection and imaging findings.",
        "Diagnosis of recurrent pneumonia confirmed; patient started on prophylactic antibiotic therapy.",
        "Recurrent pneumonia diagnosed; patient is being monitored for underlying conditions and treated accordingly.",
        "Patient diagnosed with recurrent pneumonia; treatment includes antibiotics and immune system support.",
        "Recurrent pneumonia diagnosed; the patient is under observation with long-term management plans."
    ]
}



age_list = list(range(1, 90))

report_data = report_templates.get(random.choice(diagnoses_list), [])
if report_data:
    report = random.choice(report_data)
else:
    report = "Default Report"

# Generate 10,000 rows of random data
random_data = [
    {
        "Age": random.choice(age_list),
        "Symptoms": random.sample(symptoms_list, k=random.randint(1, len(symptoms_list))),
        "Diagnosis": diagnosis,
        "Report": random.choice(report_templates.get(diagnosis, [])) if report_templates.get(diagnosis, []) else "No report available"
    }
    for _ in range(10000)
    for diagnosis in [random.choice(diagnoses_list)]  # Ensure random diagnosis per row
]

# Convert to DataFrame
df = pd.DataFrame(random_data)

# Convert list of symptoms to comma-separated string
df["Symptoms"] = df["Symptoms"].apply(lambda x: ", ".join(x))

# Display the first 5 rows to check the output
print(df.head())

# Check the balance of diagnoses
diagnosis_counts = df['Diagnosis'].value_counts()
print(diagnosis_counts)

# For underrepresented diagnoses, duplicate them in the dataset (oversampling)
min_count = diagnosis_counts.min()  # You could also choose a custom strategy to determine this
df_balanced = df.groupby('Diagnosis', group_keys=False).apply(
    lambda x: x.sample(min_count, replace=True)).reset_index(drop=True)

print(df_balanced['Diagnosis'].value_counts())

# Save to CSV
output_path = "Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_10k.csv"
df.to_csv(output_path, index=False)

output_path
