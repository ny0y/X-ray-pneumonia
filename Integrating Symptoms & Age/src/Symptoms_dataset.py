import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# Load datasets
train_data = pd.read_csv('Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_training.csv')
test_data = pd.read_csv('Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_testing.csv')

# Inspect the data structure
print(train_data.head())
print(test_data.head())

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenize training data for separate outputs
train_inputs = tokenizer(
    list(train_data['Age'].astype(str) + '; ' + train_data['Symptoms']),
    truncation=True, padding=True, return_tensors="pt", max_length=512
)
train_diagnosis_labels = tokenizer(
    list(train_data['Diagnosis']),
    truncation=True, padding=True, return_tensors="pt", max_length=512
)
train_report_labels = tokenizer(
    list(train_data['Report']),
    truncation=True, padding=True, return_tensors="pt", max_length=512
)

# Tokenize test data for separate outputs
test_inputs = tokenizer(
    list(test_data['Age'].astype(str) + '; ' + test_data['Symptoms']),
    truncation=True, padding=True, return_tensors="pt", max_length=512
)
test_diagnosis_labels = tokenizer(
    list(test_data['Diagnosis']),
    truncation=True, padding=True, return_tensors="pt", max_length=512
)
test_report_labels = tokenizer(
    list(test_data['Report']),
    truncation=True, padding=True, return_tensors="pt", max_length=512
)

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
            'labels': self.diagnosis_outputs['input_ids'][idx],  # Default labels for loss calculation        
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
    output_dir='./results',              # Directory to save results
    evaluation_strategy="epoch",       # Evaluate after each epoch
    save_strategy="epoch",             # Save checkpoint after each epoch
    load_best_model_at_end=True,         # Load the best model after training
    num_train_epochs=3,                  # Number of epochs
    per_device_train_batch_size=8,       # Batch size for training
    per_device_eval_batch_size=8,        # Batch size for evaluation
    save_total_limit=2,                  # Limit number of saved checkpoints
    logging_dir='./logs',                # Directory for logs
    logging_steps=10,                    # Log every 10 steps
    warmup_steps=500,                    # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                   # Strength of weight decay
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
    tokenizer=tokenizer
)

# Create a tqdm progress bar for the number of steps
train_steps = len(train_dataset) // training_args.per_device_train_batch_size
pbar = tqdm(total=train_steps, desc="Training Progress", ncols=100)

# Custom callback to update the progress bar
class ProgressBarCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        pbar.update(1)  # Update progress bar on each step

# Add the callback to the trainer
trainer.add_callback(ProgressBarCallback)

# Start training
trainer.train()

# Close the progress bar after training
pbar.close()

# Save the fine-tuned model
model.save_pretrained("./Integrating Symptoms & Age/synthetic data/model")
tokenizer.save_pretrained("./Integrating Symptoms & Age/synthetic data/model")
