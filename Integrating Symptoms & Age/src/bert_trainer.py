from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv('Integrating Symptoms & Age/synthetic data/dataset/symptoms_diagnoses_dataset_10k.csv')

# Prepare data
class SymptomsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        # Combine Age and Symptoms into a single string for input
        self.texts = (df['Age'].astype(str) + ' ' + df['Symptoms']).tolist()  # Concatenate age and symptoms
        self.labels = df['Diagnosis'].astype('category').cat.codes.tolist()  # Convert diagnosis to category codes
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SymptomsDataset(data, tokenizer, max_len=128)

# DataLoader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['Diagnosis'].unique()))

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop with Progress Bar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate F1 Score and Confusion Matrix
def calculate_metrics(predictions, labels):
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)
    return f1, cm

for epoch in range(3):  # Train for 3 epochs
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Store predictions and labels
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"Batch Loss": loss.item()})

    # Calculate F1 and Confusion Matrix after each epoch
    f1, cm = calculate_metrics(all_preds, all_labels)
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, F1 Score: {f1}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data['Diagnosis'].unique(), yticklabels=data['Diagnosis'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Epoch {epoch + 1}')
    plt.show()

# Save the model
model.save_pretrained('Integrating Symptoms & Age/synthetic data/model/bert_symptoms_model')
tokenizer.save_pretrained('Integrating Symptoms & Age/synthetic data/model/bert_symptoms_model')


# Prepare Test Dataset
test_data = pd.read_csv('Integrating Symptoms & Age/synthetic data/dataset/test_data.csv')
test_dataset = SymptomsDataset(test_data, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate the Model
model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []

# Disable gradient calculation for evaluation
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute F1-Score
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"F1 Score (Weighted): {f1}")

# Compute Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_data['Diagnosis'].unique(), yticklabels=test_data['Diagnosis'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
