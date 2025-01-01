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
def generate_prediction(age, symptoms):
    # Prepare the input text with explicit request for both diagnosis and report
    input_text = f"Age: {age}; Symptoms: {symptoms}. Please provide both a diagnosis and a detailed report."
    
    # Tokenize the input
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # Move the model and input tensors to the same device (GPU in this case)
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to the same device
    
    # Generate predictions for diagnosis and report
    outputs = model.generate(**tokens, max_new_tokens=100, no_repeat_ngram_size=2, do_sample=True, top_k=50)
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Handle the output and ensure clear splitting between Diagnosis and Report
    diagnosis = "No diagnosis available"
    report = "No detailed report generated. Please consult a medical professional for further analysis."

    # Look for the diagnosis and report sections in the generated text
    if "Diagnosis:" in generated_text:
        diagnosis_start = generated_text.find("Diagnosis:") + len("Diagnosis: ")
        report_start = generated_text.find("Report:") + len("Report: ")
        
        diagnosis = generated_text[diagnosis_start:report_start].strip()
        report = generated_text[report_start:].strip()

    # Get current date and time for the report
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the final report
    report_text = f"""
    Time: {current_time}
    Date: {current_time.split()[0]}  # Extract just the date part

    Report for Client:
    ------------------------------------
    - Age: {age}
    - Symptoms: {symptoms}
    - Diagnosis: {diagnosis}

    Report:
    {report}
    """
    
    return report_text

# Example test
print(generate_prediction(15, "headache, fatigue, sore throat, fever, joint pain"))
