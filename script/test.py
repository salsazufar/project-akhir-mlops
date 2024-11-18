import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import initialize_model, device  # Importing from train.py
import mlflow
import mlflow.pytorch
from mlflow import log_metric, start_run

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow/#/"
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Dataset paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
test_dir = os.path.join(base_dir, "test")

# Validate paths
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"

# Data transformation for the test set
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(test_dir, data_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
dataset_sizes = {'test': len(test_dataset)}

# Load the trained model
num_classes = 4  # Update as per your dataset
model = initialize_model(num_classes).to(device)

# Load the saved best model weights
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/best_model_weights.pth"))
assert os.path.exists(model_path), f"Model weights file not found: {model_path}"
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Test the model with MLflow logging
def computeTestSetAccuracy(model, criterion, dataloader):
    test_loss, test_corrects = 0.0, 0

    with start_run():  # Start an MLflow run
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                # Compute accuracy
                _, predictions = torch.max(outputs, 1)
                test_corrects += torch.sum(predictions == labels.data)

        # Calculate average loss and accuracy
        avg_loss = test_loss / len(dataloader.dataset)
        avg_accuracy = test_corrects.double() / len(dataloader.dataset)

        # Log metrics to MLflow
        log_metric("test_loss", avg_loss)
        log_metric("test_accuracy", avg_accuracy)

        # Print results
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")


# Run the testing phase
if __name__ == "__main__":
    test_loss, test_accuracy = computeTestSetAccuracy(model, criterion, test_loader)
