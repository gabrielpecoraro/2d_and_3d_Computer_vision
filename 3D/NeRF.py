import torch
import torch.nn as nn
import numpy as np
import os
from imageio import imread
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def positional_encoding(x, nums_frequencies=10):
    """NN struggles with high frequency inputs so we encode the inputs using sine and cos

    Args:
        x (array): position array
        nums_frequencies (int, optional): _description_. Defaults to 10.
    """
    frequencies = [2**i for i in range(nums_frequencies)]
    encoding = []

    for freq in frequencies:
        encoding.append(torch.sin(freq * x))
        encoding.append(torch.cos(freq * x))

    # Concatenate tensor accross the last dimension
    return torch.cat(encoding, dim=-1)


class NeRF(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(NeRF, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # Output [R, G, B, sigma]
        )

    def forward(self, x):
        return self.network


def rendering_ray(ray_samples, densities, colors):
    """Perform volumetric rendering along a ray

    Args:
        ray_samples (_type_): Point sampled along the ray (N, 3)
        densities (_type_): Predicted density for each point (N, 1)
        colors (_type_): Predicted RGB color for each point (N, 3)
    """
    # Compute the transmittance T = product(1 - accumulated_opacity)
    alphas = 1 - torch.exp(-densities)
    weights = alphas * torch.cumprod(
        torch.cat(
            [torch.ones(1), 1.0 - alphas[:-1] + 1e-10], dim=0
        )  # 1e-10 added to maintain numerical stability
    )
    rendered_color = torch.sum(weights[:, None] * colors, dim=0)
    return rendered_color


def load_data(data_dir: str):
    """Loading the data to feed the model

    Args:
        data_dir (str): path name
    """
    images = []
    poses = []

    # Iterate over files in the dataset folder
    for file in sorted(os.listdir(data_dir)):
        if file.endswith("png") or file.endswith("jpg"):
            # Load images
            image = imread(os.path.join(data_dir, file))
            image.append(image)

        elif file.endswith(".txt"):
            # Load poses
            pose = np.loadtxt(os.path.join(data_dir, file)).reshape(3, 4)
            poses.append(pose)

    return np.array(image), np.array(poses)


def process_dataset(images):
    """Process the dataset by normalizing it

    Args:
        images (array): dataset

    Returns:
        array: normalized dataset
    """
    images = images.astype(np.float32) / 255.0  # Normalize images
    return images


class CoarseFineNeRF(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CoarseFineNeRF, self).__init__()
        self.coarse = NeRF(input_dim, hidden_dim)
        self.fine = NeRF(input_dim, hidden_dim)

    def forward(self, x):
        # Coarse Prediction
        coarse_output = self.coarse(x)

        # Fine prediction
        fine_output = self.fine(x)
        return coarse_output, fine_output


def ray_generation(camera_matrix, image_size):
    height, width = image_size
    i, j = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )  # specify indexing convention

    # Generate directions in camera space
    directions = torch.stack(
        [
            (j - camera_matrix[0, 2]) / camera_matrix[0, 0],
            (i - camera_matrix[1, 2]) / camera_matrix[1, 1],
            torch.ones_like(i),
        ],
        dim=-1,
    )

    return directions


def loss_fn(predicted, target):
    """
    Computes the photometric loss between predicted and target pixel colors.
    Args:
        predicted: Predicted pixel values (N, 3).
        target: Ground truth pixel values (N, 3).
    Returns:
        Scalar loss value.
    """
    return F.mse_loss(predicted, target)


def sparsity_loss(densities):
    """
    Regularizes density predictions to enforce sparsity.
    Args:
        densities: Predicted density values (N, 1).
    Returns:
        Scalar sparsity loss.
    """
    return torch.mean(densities)


def sample_points_along_rays(rays, num_samples, near=2.0, far=6.0):
    """
    Sample points along rays for NeRF rendering.

    Args:
        rays: Ray directions tensor (H, W, 3) or (N, 3)
        num_samples: Number of points to sample along each ray
        near: Near bound for sampling
        far: Far bound for sampling

    Returns:
        sampled_points: Tensor of sampled 3D points (H, W, num_samples, 3) or (N, num_samples, 3)
    """
    # Create linearly spaced depth values
    t_vals = torch.linspace(near, far, num_samples, device=rays.device)

    # Add some randomness to sampling (stratified sampling)
    if rays.training:
        # Random jittering for training
        mids = 0.5 * (t_vals[1:] + t_vals[:-1])
        upper = torch.cat([mids, t_vals[-1:]], dim=0)
        lower = torch.cat([t_vals[:1], mids], dim=0)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    # Expand t_vals to match ray dimensions
    if len(rays.shape) == 3:  # (H, W, 3)
        t_vals = t_vals.expand(rays.shape[0], rays.shape[1], num_samples)
        # Compute 3D points: origin + t * direction
        # Assuming ray origin is at (0, 0, 0) - adjust if needed
        sampled_points = rays.unsqueeze(-2) * t_vals.unsqueeze(-1)
    else:  # (N, 3)
        t_vals = t_vals.expand(rays.shape[0], num_samples)
        sampled_points = rays.unsqueeze(-2) * t_vals.unsqueeze(-1)

    return sampled_points


def render_scene(nerf, camera_pose, image_size, num_samples=128):
    """
    Renders a new view using the trained NeRF model.
    Args:
        nerf: Trained NeRF model.
        camera_pose: Pose matrix for the new camera view.
        image_size: (height, width) of the output image.
        num_samples: Number of points to sample along each ray.
    Returns:
        Rendered image as a tensor (height, width, 3).
    """
    # Generate rays
    rays = ray_generation(camera_pose, image_size).to(nerf.device)

    # Sample points along rays
    sampled_points = sample_points_along_rays(rays, num_samples)

    # Pass sampled points through NeRF
    densities, colors = nerf(sampled_points)

    # Perform volumetric rendering
    rendered_image = rendering_ray(densities, colors)
    return rendered_image


if __name__ == "__main__":
    # Positional Encoding
    encoded_position = positional_encoding(coords)
    encoded_direction = positional_encoding(view_dir)

    # MLP prediction
    nerf = NeRF(encoded_position.shape[-1] + encoded_direction.shape[-1])
    predicted = nerf(torch.cat([encoded_position, encoded_direction], dim=-1))

    # Render a pixel color from densities and RGB prediction
    density, rgb = predicted[:, 3:], predicted[:, :3]
    pixel_color = rendering_ray(samples, density, rgb)

    optimizer = torch.optim.Adam(nerf.parameters, lr=1e-4)

    # Moving to GPU
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    nerf = nerf.to(device)

    # Initialize tensorboard writer

    writer = SummaryWriter()
    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            writer.add_scalar("Loss/Train", epoch_loss, epoch)
            rays, ground_truth = batch

            optimizer.zero_grad()

            predicted_colors = nerf(rays)

            loss = loss_fn(predicted_colors, ground_truth)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Render a validation view and log the image
            if epoch % 10 == 0:
                with torch.no_grad():
                    validation_image = render_scene(nerf, rays)  # Custom function
                    writer.add_image("Rendered View", validation_image, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
