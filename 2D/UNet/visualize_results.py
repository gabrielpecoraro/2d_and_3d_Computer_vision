import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from unet import UNet

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_model(model_path="unet_pets.pth"):
    """Load the trained UNet model"""
    model = UNet(in_c=3, out_c=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image_path):
    """Predict mask for a single image"""
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    return image, pred_mask, original_size


def visualize_results(model, num_images=6):
    """Visualize original images and their predicted masks"""
    # Get list of test images
    image_folder = "data/images"
    test_images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")][
        :num_images
    ]

    # Create subplot grid
    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
    fig.suptitle("UNet Segmentation Results", fontsize=16)

    for i, img_name in enumerate(test_images):
        img_path = os.path.join(image_folder, img_name)

        # Get prediction
        original_image, pred_mask, original_size = predict_image(model, img_path)

        # Plot original image
        axes[0, i].imshow(original_image)
        axes[0, i].set_title(f"Original: {img_name[:15]}...", fontsize=10)
        axes[0, i].axis("off")

        # Plot predicted mask
        axes[1, i].imshow(pred_mask, cmap="gray")
        axes[1, i].set_title("Predicted Mask", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_overlay(model, num_images=3):
    """Visualize images with mask overlay"""
    image_folder = "data/images"
    test_images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")][
        :num_images
    ]

    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    fig.suptitle("UNet Results with Overlay", fontsize=16)

    for i, img_name in enumerate(test_images):
        img_path = os.path.join(image_folder, img_name)

        # Get prediction
        original_image, pred_mask, original_size = predict_image(model, img_path)

        # Create colored overlay
        colored_mask = np.zeros((*pred_mask.shape, 3))
        colored_mask[pred_mask > 0.5] = [1, 0, 0]  # Red for foreground

        # Convert original image to numpy
        img_array = np.array(original_image.resize((256, 256))) / 255.0

        # Blend image and mask
        alpha = 0.6
        blended = alpha * img_array + (1 - alpha) * colored_mask

        axes[i].imshow(blended)
        axes[i].set_title(f"Overlay: {img_name[:15]}...", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("overlay_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def compare_with_ground_truth(model, num_images=4):
    """Compare predictions with ground truth masks if available"""
    image_folder = "data/images"
    mask_folder = "data/annotations/trimaps"

    test_images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")][
        :num_images
    ]

    fig, axes = plt.subplots(3, num_images, figsize=(4 * num_images, 12))
    fig.suptitle("Original vs Ground Truth vs Prediction", fontsize=16)

    for i, img_name in enumerate(test_images):
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name.replace(".jpg", ".png"))

        # Get prediction
        original_image, pred_mask, original_size = predict_image(model, img_path)

        # Plot original image
        axes[0, i].imshow(original_image)
        axes[0, i].set_title("Original", fontsize=10)
        axes[0, i].axis("off")

        # Plot ground truth if available
        if os.path.exists(mask_path):
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256))
            gt_array = np.array(gt_mask) / 255.0
            axes[1, i].imshow(gt_array, cmap="gray")
            axes[1, i].set_title("Ground Truth", fontsize=10)
        else:
            axes[1, i].text(0.5, 0.5, "No GT Available", ha="center", va="center")
            axes[1, i].set_title("Ground Truth", fontsize=10)
        axes[1, i].axis("off")

        # Plot prediction
        axes[2, i].imshow(pred_mask, cmap="gray")
        axes[2, i].set_title("Prediction", fontsize=10)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def save_individual_results(model, num_images=5):
    """Save individual high-quality results"""
    image_folder = "data/images"
    test_images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")][
        :num_images
    ]

    # Create results directory
    os.makedirs("results", exist_ok=True)

    for img_name in test_images:
        img_path = os.path.join(image_folder, img_name)

        # Get prediction
        original_image, pred_mask, original_size = predict_image(model, img_path)

        # Save original
        original_image.save(f"results/original_{img_name}")

        # Save mask
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(original_size)
        mask_img.save(f"results/mask_{img_name}")

        print(f"Saved results for {img_name}")


def main():
    print("Loading trained model...")
    model = load_model("unet_pets.pth")

    print("Generating visualizations...")

    # 1. Basic results visualization
    visualize_results(model, num_images=6)

    # 2. Overlay visualization
    visualize_overlay(model, num_images=3)

    # 3. Comparison with ground truth
    compare_with_ground_truth(model, num_images=4)

    # 4. Save individual results
    save_individual_results(model, num_images=5)

    print("Visualization complete! Check the generated images:")
    print("- segmentation_results.png")
    print("- overlay_results.png")
    print("- comparison_results.png")
    print("- results/ folder with individual images")


if __name__ == "__main__":
    main()
