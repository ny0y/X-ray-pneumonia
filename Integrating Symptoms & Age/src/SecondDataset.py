from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime

# Path to your trained model
model_path = "Integrating Symptoms & Age/synthetic data/model"  # Change this to the correct path of your model

# Load the custom trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)

# Ensure the model is on the correct device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate the diagnosis and report
def generate_prediction(model, tokenizer, age, symptoms):
    # Prepare the input string  
    input_text = f"{age}; {symptoms}"
    
    # Tokenize the input  
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    # Generate diagnosis  
    diagnosis_ids = model.generate(input_ids, max_length=50)
    diagnosis = tokenizer.decode(diagnosis_ids[0], skip_special_tokens=True)

    # Generate report  
    report_ids = model.generate(input_ids, max_length=150)
    report = tokenizer.decode(report_ids[0], skip_special_tokens=True)

    # Get current time and date  
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the output  
    formatted_report = f"""
    Time: {current_time.split()[1]}
    Date: {current_time.split()[0]}

    Report for Client:
    ------------------------------------
    - Age: {age}
    - Symptoms: {symptoms}
    - Diagnosis: {diagnosis}

    Report:
    {report}
    """
    
    return formatted_report

# Example usage  
age = 5  
symptoms = "mild fever"
report = generate_prediction(model, tokenizer, age, symptoms)
print(report)
