import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Function to load all image paths and their labels (0 for Normal, 1 for Pneumonia)
def load_data_from_directory(base_dir):
    """
    Load image file paths and their corresponding labels from the directory structure.
    The labels are determined based on the subfolders: 'NORMAL' -> 0, 'PNEUMONIA' -> 1.
    """
    image_paths = []
    labels = []

    # Iterate over each class folder (NORMAL, PNEUMONIA)
    for label, category in enumerate(['NORMAL', 'PNEUMONIA']):
        category_dir = os.path.join(base_dir, category)  # Path to the 'NORMAL' or 'PNEUMONIA' folder
        
        # Loop through all image files in the category folder
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)  # Get the full path of the image file
            if img_path.endswith(".jpg"):  # Check if the file is a jpg image
                image_paths.append(img_path)
                labels.append(label)
    
    return image_paths, labels

# Specify the base directory for each split
train_dir = "Pneumonia Prediction/Data/dataset/train"
test_dir = "Pneumonia Prediction/Data/dataset/test"
valid_dir = "Pneumonia Prediction/Data/dataset/valid"
valid_mini_dir = "Pneumonia Prediction/Data/dataset/valid-mini"

# Load data from directories
train_image_paths, train_labels = load_data_from_directory(train_dir)
test_image_paths, test_labels = load_data_from_directory(test_dir)
valid_image_paths, valid_labels = load_data_from_directory(valid_dir)
valid_mini_image_paths, valid_mini_labels = load_data_from_directory(valid_mini_dir)

# Define the data transformation (resizing, tensor conversion, and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (standard size for CNNs like ResNet)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Define custom Dataset class for loading chest X-ray images and labels
class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resizing, normalization)

        return image, label

# Create Dataset and DataLoader for train, test, and validation sets
train_dataset = ChestXrayDataset(train_image_paths, train_labels, transform)
test_dataset = ChestXrayDataset(test_image_paths, test_labels, transform)
valid_dataset = ChestXrayDataset(valid_image_paths, valid_labels, transform)
valid_mini_dataset = ChestXrayDataset(valid_mini_image_paths, valid_mini_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
valid_mini_loader = DataLoader(valid_mini_dataset, batch_size=32, shuffle=False)

