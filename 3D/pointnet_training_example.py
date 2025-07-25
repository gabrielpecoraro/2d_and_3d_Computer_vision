import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pointnet_implementation import (
    PointNetClassification,
    feature_transform_regularizer,
)


class DummyPointCloudDataset(Dataset):
    """Dummy dataset for demonstration"""

    def __init__(self, num_samples=1000, num_points=1024, num_classes=10):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random point cloud
        points = torch.randn(3, self.num_points)
        label = torch.randint(0, self.num_classes, (1,)).long()
        return points, label.squeeze()


def train_pointnet_classification():
    """Training function for PointNet classification"""

    # Hyperparameters
    batch_size = 16
    epochs = 10
    learning_rate = 0.001
    num_classes = 10
    reg_weight = 0.001

    # Dataset and DataLoader
    dataset = DummyPointCloudDataset(num_samples=1000, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, and optimizer
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = PointNetClassification(k=num_classes).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred, trans, trans_feat = model(data)

            # Classification loss
            loss = criterion(pred, target)

            # Add regularization for feature transformation
            if trans_feat is not None:
                reg_loss = feature_transform_regularizer(trans_feat)
                loss += reg_weight * reg_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = pred.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


def evaluate_model(model, dataloader, device):
    """Evaluation function"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            pred, _, _ = model(data)
            _, predicted = pred.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # Train the model
    print("Starting PointNet training...")
    trained_model = train_pointnet_classification()

    # Create test dataset and evaluate
    print("\nEvaluating model...")
    test_dataset = DummyPointCloudDataset(num_samples=200, num_classes=10)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy = evaluate_model(trained_model, test_dataloader, device)

    # Save the model
    torch.save(trained_model.state_dict(), "pointnet_classification.pth")
    print("Model saved as 'pointnet_classification.pth'")
