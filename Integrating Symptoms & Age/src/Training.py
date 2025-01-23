import pandas as pd  
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq  
from torch.utils.data import Dataset, DataLoader  
import torch  
from tqdm import tqdm  
from sklearn.metrics import confusion_matrix, classification_report  
import seaborn as sns  
import matplotlib.pyplot as plt
from Symptoms_Diagnosis_Report_dataset import diagnoses_list

# Load datasets  
train_data = pd.read_csv('Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_training.csv')
test_data = pd.read_csv('Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_testing.csv')

# Check data structure  
print("Training data structure:\n", train_data.head())
print("Testing data structure:\n", test_data.head())

# Ensure required columns exist
required_columns = {'Age', 'Symptoms', 'Diagnosis', 'Report'}
if not required_columns.issubset(train_data.columns) or not required_columns.issubset(test_data.columns):
    raise ValueError(f"Missing required columns! Found in training: {train_data.columns}, testing: {test_data.columns}")

# Check for missing values and handle them
print("Checking for missing values in training data:")
print(train_data.isnull().sum())
print("Checking for missing values in testing data:")
print(test_data.isnull().sum())

train_data = train_data.dropna()
test_data = test_data.dropna()

# Initialize tokenizer  
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

# Tokenize data  
def tokenize_data(data, tokenizer, max_length=512):
    try:
        inputs = tokenizer(
            list(data['Age'].astype(str) + '; ' + data['Symptoms']),
            truncation=True, padding=True, return_tensors="pt", max_length=max_length
        )
        diagnosis_labels = tokenizer(
            list(data['Diagnosis']),
            truncation=True, padding=True, return_tensors="pt", max_length=max_length
        )
        report_labels = tokenizer(
            list(data['Report']),
            truncation=True, padding=True, return_tensors="pt", max_length=max_length
        )
        return inputs, diagnosis_labels, report_labels
    except Exception as e:
        print(f"Error during tokenization: {e}")
        print(data.head())
        raise

# Tokenize the datasets  
train_inputs, train_diagnosis_labels, train_report_labels = tokenize_data(train_data, tokenizer)
test_inputs, test_diagnosis_labels, test_report_labels = tokenize_data(test_data, tokenizer)

# Custom dataset class  
class SymptomsDataset(Dataset):
    def __init__(self, inputs, diagnosis_outputs, report_outputs):
        self.inputs = inputs  
        self.diagnosis_outputs = diagnosis_outputs  
        self.report_outputs = report_outputs

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.diagnosis_outputs['input_ids'][idx],
            'decoder_input_ids': self.report_outputs['input_ids'][idx]
        }

# Create dataset instances  
train_dataset = SymptomsDataset(train_inputs, train_diagnosis_labels, train_report_labels)
test_dataset = SymptomsDataset(test_inputs, test_diagnosis_labels, test_report_labels)

# Load the T5 model  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

# Define training arguments  
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",  # Match evaluation strategy with save strategy
    eval_steps=100,         # Evaluate every 100 steps
    save_strategy="steps",  # Save every 100 steps
    save_steps=100,         # Ensure saving aligns with eval steps
    load_best_model_at_end=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Enable mixed-precision training for speed
)

# Data collator for dynamic padding  
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize trainer  
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training  
trainer.train()

# Save the fine-tuned model  
model.save_pretrained("./Integrating Symptoms & Age/synthetic data/model")
tokenizer.save_pretrained("./Integrating Symptoms & Age/synthetic data/model")

# Evaluation: Generate predictions
predictions = []
actual_diagnoses = []

for batch in DataLoader(test_dataset, batch_size=16):  # Batch size for evaluation
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50)
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        actual_diagnoses.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

# Create a confusion matrix  
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Define classes based on actual diagnoses  
classes = sorted(set(diagnoses_list))  # Unique diagnosis labels

# Plot the confusion matrix  
plot_confusion_matrix(actual_diagnoses, predictions, classes)

# Print the classification report  
print("\nClassification Report:\n")
print(classification_report(actual_diagnoses, predictions, target_names=classes))

# Print out some examples to see how well the model performed  
for i in range(5):  # Print first 5 predictions for comparison  
    print(f"Input Age/Symptoms: {tokenizer.decode(test_dataset[i]['input_ids'], skip_special_tokens=True)}")  
    print(f"Predicted Diagnosis: {predictions[i]}")
    print(f"Actual Diagnosis: {actual_diagnoses[i]}")
    print("-" * 50)
