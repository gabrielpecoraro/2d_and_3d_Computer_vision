import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2

from typing import Annotated
from annotated_types import Gt

print(torch.mps.is_available())


# use torch.cuda.is_available for CUDA GPU


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),  # YOLO input size
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
)

# Download and load VOC dataset (much smaller and easier!)
train_dataset = VOCDetection(
    root="../data/YOLO",
    year="2012",
    image_set="train",
    download=True,
    transform=transform,
)

val_dataset = VOCDetection(
    root="../data/YOLO",
    year="2012",
    image_set="val",
    download=True,
    transform=transform,
)

# VOC has 20 classes
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class ConvBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        self.relu(x)
        return x


class YOLOBackbone(nn.Module):
    # Model backbone
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((2, 2)),
        )

    def forward(self, x):
        return self.layers(x)


class YOLODetectHead(nn.Module):
    # Classifier, Bounding Boxes, confidence scores
    def __init__(self, grid_size, num_classes, num_anchors):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.detector(x).permute(0, 2, 3, 1).contiguous()


class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
        super().__init__()
        self.backbone = YOLOBackbone()
        self.detect = YOLODetectHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detect(features)
        return predictions


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        label_path = os.path.join(
            self.label_dir, self.images[index].replace(".jpg", ".txt")
        )

        # Load Images
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        if self.transforms:
            image = self.transforms(image)


def generate_anchors(scales, ratios):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return anchors


def convert_to_yolo_format(width, height, bbox):
    """Converts absolute bounding box to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return [x_center, y_center, box_width, box_height]


def yolo_loss(predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    # Unpack predictions and target
    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]
    target_boxes = targets[..., :4]
    target_conf = predictions[..., 4]
    target_classes = predictions[..., 5:]

    # Localization loss
    box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

    # Confidence loss
    obj_loss = torch.sum((pred_conf - target_conf) ** 2)
    noobj_loss = lambda_noobj * torch((pred_conf[target_conf == 0]) ** 2)

    # Clasification loss
    class_loss = torch.sum((pred_classes[target_classes == 0]) ** 2)

    loss = box_loss + obj_loss + noobj_loss + class_loss

    return loss


def train(train_loader, num_epochs: Annotated[int, Gt(0)]):
    model = YOLO(grid_size=7, num_classes=20, num_anchors=3)
    criterion = yolo_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in train_loader:
            # Chech GPU
            if torch.cuda.is_available():
                print("Moved to CUDA")
                images = images.to("cuda")
                targets = targets.to("cuda")
            elif torch.mps.is_available():
                print("Moved to MPS")
                images = images.to("mps")
                targets = targets.to("mps")

            # Forward pass
            predictions = model(images)

            # Loss Function
            loss = criterion(predictions, targets, num_classes=20)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union


if __name__ == "__main__":
    """train_dataset = YOLODataset(
        img_dir="", label_dir="data/labels", transforms=transforms.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
"""
