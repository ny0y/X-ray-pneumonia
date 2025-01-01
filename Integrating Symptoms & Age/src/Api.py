from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the app
app = FastAPI()

# Request body model
class PredictRequest(BaseModel):
    age: int
    symptoms: str

@app.post("/predict")
def predict(request: PredictRequest):
    input_text = f"Age: {request.age}; Symptoms: {request.symptoms}"
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"Prediction": response}

# Run the API using uvicorn
# Save this script as `app.py` and run: `uvicorn app:app --reload`
