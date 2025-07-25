import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import requests
from unet import UNet

# Set device for Apple Silicon GPU and CUDA
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple Silicon GPU) is available!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available!")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Using device: {device}")


# Download Oxford-IIIT Pet Dataset
def download_dataset():
    """Download and extract the Oxford-IIIT Pet Dataset"""
    if not os.path.exists("data"):
        os.makedirs("data")

    # URLs for the dataset
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    masks_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    print("Downloading dataset... (this may take a while)")

    # Download images
    if not os.path.exists("data/images.tar.gz"):
        response = requests.get(images_url)
        with open("data/images.tar.gz", "wb") as f:
            f.write(response.content)

    # Download annotations (masks)
    if not os.path.exists("data/annotations.tar.gz"):
        response = requests.get(masks_url)
        with open("data/annotations.tar.gz", "wb") as f:
            f.write(response.content)

    # Extract files
    import tarfile

    if not os.path.exists("data/images"):
        with tarfile.open("data/images.tar.gz", "r:gz") as tar:
            tar.extractall("data/")

    if not os.path.exists("data/annotations"):
        with tarfile.open("data/annotations.tar.gz", "r:gz") as tar:
            tar.extractall("data/")

    print("Dataset downloaded and extracted!")


# Custom Dataset Class
class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get list of image files
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

        # Limit to max_samples if specified
        if max_samples:
            self.images = self.images[:max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load mask (segmentation map)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # Create dummy mask if not found
            mask = Image.new("L", image.size, 0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask


# Training function
def train_model():
    # Download dataset
    download_dataset()

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Create dataset
    dataset = PetDataset(
        image_dir="data/images",
        mask_dir="data/annotations/trimaps",
        transform=transform,
        max_samples=500,  # Limit to 500 samples
    )

    print(f"Dataset size: {len(dataset)} samples")
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders (increased batch size for better GPU utilization)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize model on MPS
    model = UNet(in_c=3, out_c=1).to(device)
    print(f"Model moved to {device}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move data to MPS
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print("-" * 40)

    # Save model
    torch.save(model.state_dict(), "unet_pets.pth")
    print("Model saved as 'unet_pets.pth'")

    return model


# Test the trained model
def test_model(model_path="unet_pets.pth"):
    # Load model on MPS
    model = UNet(in_c=3, out_c=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")

    # Test on a single image
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Load test image
    test_images = os.listdir("data/images")[:5]  # Test on first 5 images

    for img_name in test_images:
        img_path = os.path.join("data/images", img_name)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Save prediction
        pred_img = Image.fromarray(pred_mask)
        pred_img.save(f"prediction_{img_name}")
        print(f"Saved prediction for {img_name}")


if __name__ == "__main__":
    print("Starting training...")
    model = train_model()

    print("Testing model...")
    test_model()

    print("Training and testing complete!")
