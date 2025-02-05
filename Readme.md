
Environment Configuration

.env: Ensure this file contains the necessary environment variables such as API keys, paths to datasets, or configuration settings for your project. For example, you might store paths to your datasets or model checkpoints in this file for easy access across your scripts.
Integrating Symptoms & Age:

This folder appears to handle the integration of features such as symptoms and age into the prediction or classification model.
bert_trainer.py: This script is likely where you train a BERT-based model. Ensure that you handle data preprocessing, model training, and evaluation within this script.
Symptoms_Diagnosis_Report_dataset.py: This script might be responsible for managing the dataset that links symptoms to diagnoses and generating reports.
SecondDataset.py: This script likely processes another dataset or acts as an auxiliary for the main dataset. Make sure the preprocessing in this script aligns with the requirements of the primary dataset.
Synthetic Data:

dataset/: This is where your synthetic and real datasets will be stored.
You can generate synthetic data (combining symptoms and diagnoses) in the dataset/ folder if you are simulating different conditions or augmenting your dataset. Ensure the synthetic data mimics the real-world distribution of symptoms and diagnoses.
model/: This folder could store your models, either the pre-trained ones or the ones you develop. Make sure your models are saved with clear names indicating the experiment or approach used.
LLM-based Report Generation:

This directory seems focused on generating reports using a large language model (LLM).
Depending on your implementation, you may need to integrate an API or library (like GPT) for report generation, depending on your goal. If using a transformer-based model for report generation, consider saving the models and checkpoints here.
Pneumonia Prediction:

This section seems dedicated to the core functionality of predicting pneumonia from X-ray images.
Ensure your data pipeline and model architecture are aligned to handle image data for training and testing. The model should focus on distinguishing between normal and pneumonia X-rays.
Dataset:

The dataset is categorized into train, test, and validation folders, which is great for model training, evaluation, and testing.
Make sure that images in the respective folders are labeled correctly (e.g., NORMAL, PNEUMONIA).
Mini datasets (valid-mini) are helpful for quick model checks but ensure they are representative of the full dataset for validation.
Model Trained:

This folder holds the trained model, specifically the chest_xray_model.pth.
After training the model, save it in this folder and load it during evaluation or inference for testing predictions on new X-ray images.
Source Code:

dataloader.py: This script will handle loading and preprocessing image data from the dataset. Ensure you use a robust data augmentation pipeline to improve model generalization.
evaluate_model.py: After training, this script can be used to evaluate the model on test or validation data. It could include metrics like accuracy, precision, recall, F1 score, etc.
train_model.py: This is the core script for training the pneumonia prediction model. Make sure to define the architecture (e.g., CNN, ResNet) and training pipeline (loss function, optimizer, etc.) here.
Test Images:

The Test_JPG/ folder holds test images you can use to evaluate the model in real-time or during post-training validation. Ensure these images are correctly labeled and preprocessed to match the model's input requirements.
Readme.md:

The Readme.md file should outline the purpose of your project, how to set up the environment, dependencies, and how to use the different components (e.g., training, testing, evaluation). It should also provide any other relevant information for collaborators or users.
requirements.txt:

This file lists all the dependencies your project requires. Ensure you list packages like torch, torchvision, pandas, scikit-learn, and any other libraries essential for your model and data processing.



X-RAY PNEUMONIA/
├── .env                                      # Environment configuration file
├── Integrating Symptoms & Age/              # Feature or module folder
|   ├── src/
│   |   ├── bert_trainer.py
│   |   ├── Symptoms_Diagnosis_Report_dataset.py
│   |   └── SecondDataset.py                      # Python script for handling a secondary dataset
|   └──synthetic data/
│       ├── dataset/                              # Placeholder for synthetic dataset files
|   └──synthetic data/
│       ├── dataset/                              # Placeholder for synthetic dataset files
│       └── model/                                # Placeholder for synthetic model files
├── LLM-based Report Generation/             # Placeholder for report generation functionality
├── Pneumonia Prediction/                    # Placeholder for Pneumonia Prediction functionality
│   ├── dataset/                                 # Dataset folder
│   ├── test/                                # Test data
│   │   ├── NORMAL/                          # Normal X-ray images for testing
│   │   └── PNEUMONIA/                       # Pneumonia X-ray images for testing
│   ├── train/                               # Training data
│   │   ├── NORMAL/                          # Normal X-ray images for training
│   │   └── PNEUMONIA/                       # Pneumonia X-ray images for training
│   ├── valid/                               # Validation data
│   │   ├── NORMAL/                          # Normal X-ray images for validation
│   │   └── PNEUMONIA/                       # Pneumonia X-ray images for validation
│   ├── valid-mini/                          # Mini validation dataset
│   │    ├── NORMAL/                          # Normal mini validation data
│   │    └── PNEUMONIA/                       # Pneumonia mini validation data
│   ├── model trained/
│   │    └── chest_xray_model.pth                 # Trained model file (PyTorch format)
│   └──src/                                     # Source code directory
│        ├── dataloader.py                        # Script for loading datasets
│        ├── evaluate_model.py                    # Script for evaluating the trained model
│        └── train_model.py                       # Script for training the model
│   ├── valid-mini/                          # Mini validation dataset
│   │    ├── NORMAL/                          # Normal mini validation data
│   │    └── PNEUMONIA/                       # Pneumonia mini validation data
│   ├── model trained/
│   │    └── chest_xray_model.pth                 # Trained model file (PyTorch format)
│   └──src/                                     # Source code directory
│        ├── dataloader.py                        # Script for loading datasets
│        ├── evaluate_model.py                    # Script for evaluating the trained model
│        └── train_model.py                       # Script for training the model
├── Test_JPG/                                # Placeholder for test image files
├── Readme.md
└── requirements.txt                         # List of dependencies





How to setup 

1- Add the Datasets in Integrating Symptoms & Age

X-RAY PNEUMONIA/
│                                     
└── Integrating Symptoms & Age/              
    ├── src/
    |   ├── Symptoms_Diagnosis_Report_dataset.py
    |   └── SecondDataset.py
    └──synthetic data/
        └── dataset/                       # Add the Dataset here
            ├── synthetic_data_train.csv   # all of these csv are added from the src of Integrating Symptoms & Age  
            ├── synthetic_data_test.csv
            ├── synthetic_data_val.csv                           
            └── symptoms_diagnoses_dataset_10k.csv 


2- Add the Datasets in Pneumonia Prediction
 
X-RAY PNEUMONIA/
│                                     
├── Pneumonia Prediction/                    # Placeholder for Pneumonia Prediction functionality
│   ├── dataset/                                 # Dataset folder
│   │   ├── test/                                # Test data
│   │   ├── NORMAL/                          # Normal X-ray images for testing
│   │   └── PNEUMONIA/                       # Pneumonia X-ray images for testing
│   ├── train/                               # Training data
│   │   ├── NORMAL/                          # Normal X-ray images for training
│   │   └── PNEUMONIA/                       # Pneumonia X-ray images for training
│   ├── valid/                               # Validation data
│   │   ├── NORMAL/                          # Normal X-ray images for validation
│   │   └── PNEUMONIA/                       # Pneumonia X-ray images for validation
│   └── valid-mini/                          # Mini validation dataset
│       ├── NORMAL/                          # Normal mini validation data
│       └── PNEUMONIA/                       # Pneumonia mini validation data
└── model trained/
    └── chest_xray_model.pth                 # Trained model file (PyTorch format)



------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                      # Pneumonia Prediction: Python Modules

## datasetloader.py

This module handles data loading and preprocessing for training, validation, and testing a model to classify chest X-ray images into `NORMAL` or `PNEUMONIA` categories.

### Key Components

1. **Function `load_data_from_directory(base_dir)`**:
   - Iterates through subdirectories (`NORMAL` and `PNEUMONIA`) to gather image file paths and their corresponding labels (`0` for `NORMAL`, `1` for `PNEUMONIA`).

2. **Paths to Datasets**:
   - `train_dir`, `test_dir`, `valid_dir`, `valid_mini_dir` specify paths to the training, testing, validation, and mini validation datasets respectively.

3. **Data Transformation**:
   - Applies resizing, tensor conversion, and normalization using `transforms.Compose`.

4. **Class `ChestXrayDataset`**:
   - A custom PyTorch `Dataset` for loading images and labels.

5. **Dataloader Initialization**:
   - Creates PyTorch `DataLoader` objects for train, validation, test, and mini-validation datasets with a batch size of `32`.

### Example Usage
```python
from datasetloader import train_loader
for inputs, labels in train_loader:
    print(inputs.shape, labels.shape)
```

---

## train_model.py

This module trains a ResNet-18 model for binary classification of chest X-ray images.

### Key Components

1. **Model Initialization**:
   - Uses a pretrained ResNet-18 model.
   - Modifies the fully connected (fc) layer for binary classification (`2` output classes).

2. **Training Loop**:
   - Trains the model for `10` epochs using the Adam optimizer and CrossEntropy loss.
   - Tracks and prints loss and accuracy for each epoch.

3. **Validation Loop**:
   - Evaluates model performance on the validation set.
   - Saves the best model based on validation accuracy.

4. **Testing Loop**:
   - Evaluates the best model on the test set.
   - Generates and displays a confusion matrix and classification report.

5. **Model Saving**:
   - Saves the trained model weights in `chest_xray_model.pth`.

### Example Usage
```python
# Train and evaluate the model
python train_model.py
```

---

## evaluate_model.py

This module evaluates a trained ResNet-18 model on new or unseen data.

### Key Components

1. **Transformation**:
   - Applies resizing, tensor conversion, and normalization consistent with training preprocessing.

2. **Model Loading**:
   - Loads saved model weights (`chest_xray_model.pth`).
   - Configures the model for inference on GPU or CPU.

3. **Prediction Functionality**:
   - Processes an input X-ray image for prediction.
   - Outputs the predicted label (`NORMAL` or `PNEUMONIA`).

### Example Usage
```python
from evaluate_model import model, transform
from PIL import Image

image = Image.open("path/to/xray.jpg")
processed_image = transform(image).unsqueeze(0).to(device)
prediction = model(processed_image)
print("Predicted Label:", torch.argmax(prediction, dim=1).item())

Epoch 1/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:32<00:00,  3.96it/s]
Epoch 1 - Loss: 0.1753, Accuracy: 0.9399
Validation - Loss: 0.1106, Accuracy: 0.9562
Epoch 2/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.22it/s] 
Epoch 2 - Loss: 0.0912, Accuracy: 0.9684
Validation - Loss: 0.1060, Accuracy: 0.9639
Epoch 3/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.14it/s] 
Epoch 3 - Loss: 0.0720, Accuracy: 0.9760
Validation - Loss: 0.0905, Accuracy: 0.9708
Epoch 4/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.19it/s] 
Epoch 4 - Loss: 0.0728, Accuracy: 0.9747
Validation - Loss: 0.1906, Accuracy: 0.9305
Epoch 5/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.25it/s] 
Epoch 5 - Loss: 0.0703, Accuracy: 0.9738
Validation - Loss: 0.1874, Accuracy: 0.9356
Epoch 6/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.13it/s] 
Epoch 6 - Loss: 0.0482, Accuracy: 0.9801
Validation - Loss: 0.0849, Accuracy: 0.9657
Epoch 7/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.21it/s] 
Epoch 7 - Loss: 0.0526, Accuracy: 0.9828
Validation - Loss: 0.1713, Accuracy: 0.9459
Epoch 8/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.24it/s] 
Epoch 8 - Loss: 0.0361, Accuracy: 0.9858
Validation - Loss: 0.1101, Accuracy: 0.9631
Epoch 9/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.20it/s] 
Epoch 9 - Loss: 0.0279, Accuracy: 0.9895
Validation - Loss: 0.1077, Accuracy: 0.9665
Epoch 10/10 - Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.17it/s] 
Epoch 10 - Loss: 0.0693, Accuracy: 0.9755
Validation - Loss: 0.3029, Accuracy: 0.8970
Test - Loss: 0.4080, Accuracy: 0.8694



Classification Report:
              precision    recall  f1-score   support

      NORMAL       0.99      0.56      0.72       171
   PNEUMONIA       0.85      1.00      0.92       411

    accuracy                           0.87       582
   macro avg       0.92      0.78      0.82       582
weighted avg       0.89      0.87      0.86       582





------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                                  # Integrating Symptoms & Age: Python Modules

## Symptoms_Diagnosis_Report_dataset.py

This module generates synthetic data that integrates patient age, symptoms, diagnoses, and diagnosis reports. The output is a structured dataset saved as a CSV file.

### Key Components

1. **Lists of Symptoms and Diagnoses**:
   - `symptoms_list`: A comprehensive list of possible symptoms.
   - `diagnoses_list`: Includes different pneumonia and respiratory-related diagnoses.

2. **Report Templates**:
   - `report_templates`: Maps diagnoses to pre-defined textual diagnosis reports.

3. **Diagnosis-to-Symptoms Mapping**:
   - `diagnosis_to_symptoms`: Links specific symptoms to diagnoses.

4. **Synthetic Data Generation**:
   - Generates random data entries of age, symptoms, diagnosis, and corresponding reports.
   - Balances the dataset by oversampling underrepresented diagnoses.

5. **Output**:
   - Saves the generated dataset to a CSV file: `symptoms_diagnoses_dataset_10k.csv`.

### Example Usage
```python
python Symptoms_Diagnosis_Report_dataset.py
# Generates and saves a synthetic dataset.
```

---

## bert_trainer.py

This module trains a multi-task BERT model for predicting diagnoses and generating reports based on patient symptoms and age.

### Key Components

1. **Dataset Class `SymptomsDataset`**:
   - Prepares the dataset by tokenizing textual input (age + symptoms).
   - Encodes diagnoses and reports as labels.
   - Provides mappings for decoding predictions.

2. **Multi-Task BERT Model**:
   - `MultiTaskBERT`: A model with separate classification heads for diagnoses and reports.

3. **Training and Evaluation Functions**:
   - `train_epoch`: Trains the model and tracks metrics using a progress bar.
   - `evaluate_model`: Evaluates model performance and displays confusion matrices for diagnoses and reports.

4. **Main Execution**:
   - Splits the synthetic dataset into training and validation sets.
   - Trains the multi-task BERT model for 3 epochs and saves the best model.

5. **Outputs**:
   - Confusion matrices for diagnoses and reports.
   - The trained model: `best_multitask_bert_model.pth`.

### Example Usage
```bash
python bert_trainer.py
# Trains the model on the synthetic dataset.
```





------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                      # Pneumonia Prediction: Python Modules

## datasetloader.py

This module handles data loading and preprocessing for training, validation, and testing a model to classify chest X-ray images into `NORMAL` or `PNEUMONIA` categories.

### Key Components

1. **Function `load_data_from_directory(base_dir)`**:
   - Iterates through subdirectories (`NORMAL` and `PNEUMONIA`) to gather image file paths and their corresponding labels (`0` for `NORMAL`, `1` for `PNEUMONIA`).

2. **Paths to Datasets**:
   - `train_dir`, `test_dir`, `valid_dir`, `valid_mini_dir` specify paths to the training, testing, validation, and mini validation datasets respectively.

3. **Data Transformation**:
   - Applies resizing, tensor conversion, and normalization using `transforms.Compose`.

4. **Class `ChestXrayDataset`**:
   - A custom PyTorch `Dataset` for loading images and labels.

5. **Dataloader Initialization**:
   - Creates PyTorch `DataLoader` objects for train, validation, test, and mini-validation datasets with a batch size of `32`.

### Example Usage
```python
from datasetloader import train_loader
for inputs, labels in train_loader:
    print(inputs.shape, labels.shape)
```

---

## train_model.py

This module trains a ResNet-18 model for binary classification of chest X-ray images.

### Key Components

1. **Model Initialization**:
   - Uses a pretrained ResNet-18 model.
   - Modifies the fully connected (fc) layer for binary classification (`2` output classes).

2. **Training Loop**:
   - Trains the model for `10` epochs using the Adam optimizer and CrossEntropy loss.
   - Tracks and prints loss and accuracy for each epoch.

3. **Validation Loop**:
   - Evaluates model performance on the validation set.
   - Saves the best model based on validation accuracy.

4. **Testing Loop**:
   - Evaluates the best model on the test set.
   - Generates and displays a confusion matrix and classification report.

5. **Model Saving**:
   - Saves the trained model weights in `chest_xray_model.pth`.

### Example Usage
```python
# Train and evaluate the model
python train_model.py
```

---

## evaluate_model.py

This module evaluates a trained ResNet-18 model on new or unseen data.

### Key Components

1. **Transformation**:
   - Applies resizing, tensor conversion, and normalization consistent with training preprocessing.

2. **Model Loading**:
   - Loads saved model weights (`chest_xray_model.pth`).
   - Configures the model for inference on GPU or CPU.

3. **Prediction Functionality**:
   - Processes an input X-ray image for prediction.
   - Outputs the predicted label (`NORMAL` or `PNEUMONIA`).

### Example Usage
```python
from evaluate_model import model, transform
from PIL import Image

image = Image.open("path/to/xray.jpg")
processed_image = transform(image).unsqueeze(0).to(device)
prediction = model(processed_image)
print("Predicted Label:", torch.argmax(prediction, dim=1).item())

Epoch 1/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:32<00:00,  3.96it/s]
Epoch 1 - Loss: 0.1753, Accuracy: 0.9399
Validation - Loss: 0.1106, Accuracy: 0.9562
Epoch 2/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.22it/s] 
Epoch 2 - Loss: 0.0912, Accuracy: 0.9684
Validation - Loss: 0.1060, Accuracy: 0.9639
Epoch 3/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.14it/s] 
Epoch 3 - Loss: 0.0720, Accuracy: 0.9760
Validation - Loss: 0.0905, Accuracy: 0.9708
Epoch 4/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.19it/s] 
Epoch 4 - Loss: 0.0728, Accuracy: 0.9747
Validation - Loss: 0.1906, Accuracy: 0.9305
Epoch 5/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.25it/s] 
Epoch 5 - Loss: 0.0703, Accuracy: 0.9738
Validation - Loss: 0.1874, Accuracy: 0.9356
Epoch 6/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.13it/s] 
Epoch 6 - Loss: 0.0482, Accuracy: 0.9801
Validation - Loss: 0.0849, Accuracy: 0.9657
Epoch 7/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.21it/s] 
Epoch 7 - Loss: 0.0526, Accuracy: 0.9828
Validation - Loss: 0.1713, Accuracy: 0.9459
Epoch 8/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.24it/s] 
Epoch 8 - Loss: 0.0361, Accuracy: 0.9858
Validation - Loss: 0.1101, Accuracy: 0.9631
Epoch 9/10 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.20it/s] 
Epoch 9 - Loss: 0.0279, Accuracy: 0.9895
Validation - Loss: 0.1077, Accuracy: 0.9665
Epoch 10/10 - Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:30<00:00,  4.17it/s] 
Epoch 10 - Loss: 0.0693, Accuracy: 0.9755
Validation - Loss: 0.3029, Accuracy: 0.8970
Test - Loss: 0.4080, Accuracy: 0.8694



Classification Report:
              precision    recall  f1-score   support

      NORMAL       0.99      0.56      0.72       171
   PNEUMONIA       0.85      1.00      0.92       411

    accuracy                           0.87       582
   macro avg       0.92      0.78      0.82       582
weighted avg       0.89      0.87      0.86       582





------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                                  # Integrating Symptoms & Age: Python Modules

## Symptoms_Diagnosis_Report_dataset.py

This module generates synthetic data that integrates patient age, symptoms, diagnoses, and diagnosis reports. The output is a structured dataset saved as a CSV file.

### Key Components

1. **Lists of Symptoms and Diagnoses**:
   - `symptoms_list`: A comprehensive list of possible symptoms.
   - `diagnoses_list`: Includes different pneumonia and respiratory-related diagnoses.

2. **Report Templates**:
   - `report_templates`: Maps diagnoses to pre-defined textual diagnosis reports.

3. **Diagnosis-to-Symptoms Mapping**:
   - `diagnosis_to_symptoms`: Links specific symptoms to diagnoses.

4. **Synthetic Data Generation**:
   - Generates random data entries of age, symptoms, diagnosis, and corresponding reports.
   - Balances the dataset by oversampling underrepresented diagnoses.

5. **Output**:
   - Saves the generated dataset to a CSV file: `symptoms_diagnoses_dataset_10k.csv`.

### Example Usage
```python
python Symptoms_Diagnosis_Report_dataset.py
# Generates and saves a synthetic dataset.
```

---

## bert_trainer.py

This module trains a multi-task BERT model for predicting diagnoses and generating reports based on patient symptoms and age.

### Key Components

1. **Dataset Class `SymptomsDataset`**:
   - Prepares the dataset by tokenizing textual input (age + symptoms).
   - Encodes diagnoses and reports as labels.
   - Provides mappings for decoding predictions.

2. **Multi-Task BERT Model**:
   - `MultiTaskBERT`: A model with separate classification heads for diagnoses and reports.

3. **Training and Evaluation Functions**:
   - `train_epoch`: Trains the model and tracks metrics using a progress bar.
   - `evaluate_model`: Evaluates model performance and displays confusion matrices for diagnoses and reports.

4. **Main Execution**:
   - Splits the synthetic dataset into training and validation sets.
   - Trains the multi-task BERT model for 3 epochs and saves the best model.

5. **Outputs**:
   - Confusion matrices for diagnoses and reports.
   - The trained model: `best_multitask_bert_model.pth`.

### Example Usage
```bash
python bert_trainer.py
# Trains the model on the synthetic dataset.
```





------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
