import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Transformation Network (T-Net) for spatial/feature transformation"""

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        # Shared MLPs
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        # Shared MLPs with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling to get global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity transformation
        identity = torch.eye(self.k, dtype=x.dtype, device=x.device)
        identity = identity.view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity

        # Reshape to transformation matrix
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """PointNet encoder that extracts global and per-point features"""

    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        # Input transformation
        self.stn = TNet(k=3)

        # First set of shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transformation
        if self.feature_transform:
            self.fstn = TNet(k=64)

        # Second set of shared MLPs
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]

        # Input transformation
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # First MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        # Second MLPs
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetClassification(nn.Module):
    """PointNet for 3D object classification"""

    def __init__(self, k=40, dropout=0.3):
        super(PointNetClassification, self).__init__()
        self.k = k

        # Encoder
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetSegmentation(nn.Module):
    """PointNet for point cloud segmentation"""

    def __init__(self, k=13):
        super(PointNetSegmentation, self).__init__()
        self.k = k

        # Encoder (returns both global and per-point features)
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True)

        # Segmentation head
        self.conv1 = nn.Conv1d(1088, 512, 1)  # 1024 + 64 = 1088
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(x.size(0), x.size(1), self.k)

        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    """Regularization loss for feature transformation matrix"""
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


# Example usage and testing
if __name__ == "__main__":
    # Test classification model
    print("Testing PointNet Classification...")
    model_cls = PointNetClassification(k=10)
    points = torch.randn(4, 3, 1024)  # batch_size=4, 1024 points with xyz
    pred, trans, trans_feat = model_cls(points)
    print(f"Classification output shape: {pred.shape}")

    # Test segmentation model
    print("\nTesting PointNet Segmentation...")
    model_seg = PointNetSegmentation(k=13)
    pred, trans, trans_feat = model_seg(points)
    print(f"Segmentation output shape: {pred.shape}")

    # Test regularization loss
    if trans_feat is not None:
        reg_loss = feature_transform_regularizer(trans_feat)
        print(f"Regularization loss: {reg_loss.item()}")
