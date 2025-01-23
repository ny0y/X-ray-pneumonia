from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime

# Path to your trained model
model_path = "Integrating Symptoms & Age/synthetic data/model"

# Load the custom trained model and tokenizer
try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Ensure the model is on the correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate output from the model
def generate_output(model, tokenizer, input_text, max_length, do_sample=False, temperature=1.0):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=max_length, do_sample=do_sample, temperature=temperature)
    
    # Log the raw model output tokens for debugging
    print(f"Raw model output tokens: {output_ids}")
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Function to format the report
def format_report(age, symptoms, diagnosis, report):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
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

# Function to generate prediction and report
def generate_prediction(model, tokenizer, age, symptoms):
    # Diagnosis prompt with more detail
    input_text_diagnosis = f"[TASK: DIAGNOSIS] Given a child age of {age} years with symptoms of {symptoms}, generate a medical diagnosis considering common childhood diseases and infections."
    print(f"Input to model for diagnosis: {input_text_diagnosis}")
    diagnosis = generate_output(model, tokenizer, input_text_diagnosis, max_length=50, do_sample=True, temperature=1.2)
    
    # Report generation with explicit task details
    input_text_report = f"[TASK: REPORT] Based on the diagnosis of {diagnosis}, generate a detailed medical report. Include the causes, treatments, prevention, and any other relevant details. Symptoms: {symptoms}, Age: {age}."
    print(f"Input to model for report: {input_text_report}")
    report = generate_output(model, tokenizer, input_text_report, max_length=150, do_sample=True, temperature=1.2)
    
    print(f"Model output for diagnosis: {diagnosis}")
    print(f"Model output for report: {report}")
    
    return format_report(age, symptoms, diagnosis, report)

# Example usage
if __name__ == "__main__":
    age = 5
    symptoms = "mild fever"
    
    try:
        report = generate_prediction(model, tokenizer, age, symptoms)
        print(report)
    except Exception as e:
        print(f"Error generating prediction: {e}")
