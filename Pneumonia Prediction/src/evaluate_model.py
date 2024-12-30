import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Resizing the image to 224x224 (input size for ResNet)
    transforms.ToTensor(),                # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizing with ImageNet stats
])

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # We do not use pretrained weights now
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modify for binary classification
model.load_state_dict(torch.load("Pneumonia Prediction/Data/model trained/chest_xray_model.pth"))
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to make predictions on a single image
def predict_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
    
    # Map the predicted class to label
    labels = ["NORMAL", "PNEUMONIA"]
    prediction = labels[predicted.item()]
    
    # Display the image and prediction
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

    return prediction

# usage
image_path = "P.jpg"  # Replace with the path image wanna predict
prediction = predict_image(image_path)
print(f"Predicted class: {prediction}")

# usage
image_path = "N.jpg"  # Replace with the path image wanna predict
prediction = predict_image(image_path)
print(f"Predicted class: {prediction}")
