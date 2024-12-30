import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datasetloader import train_loader, valid_loader, test_loader  # Import DataLoader from datasetloader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize model, optimizer, and loss function
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify the last fully connected layer for binary classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize variables to track the best model
best_val_accuracy = 0.0
best_model_wts = None

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero out previous gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    val_epoch_loss = val_loss / len(valid_loader)
    val_epoch_accuracy = val_correct_preds / val_total_preds
    print(f"Validation - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if val_epoch_accuracy > best_val_accuracy:
        best_val_accuracy = val_epoch_accuracy
        best_model_wts = model.state_dict()  # Save the state_dict of the best model

# After training, load the best model weights
model.load_state_dict(best_model_wts)

# Testing loop
model.eval()  # Set model to evaluation mode for testing
test_loss = 0.0
test_correct_preds = 0
test_total_preds = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_correct_preds += (predicted == labels).sum().item()
        test_total_preds += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_epoch_loss = test_loss / len(test_loader)
test_epoch_accuracy = test_correct_preds / test_total_preds
print(f"Test - Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_accuracy:.4f}")

# Generate Confusion Matrix and Classification Report
cm = confusion_matrix(all_labels, all_preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "PNEUMONIA"])
cm_display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

# Save the best model
torch.save(model.state_dict(), "Image Processing/hest-xray-classification/model trained/chest_xray_model.pth")
