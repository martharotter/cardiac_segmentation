#!/usr/bin/env python3
"""
Corrected evaluation script for 2D+Time CINE segmentation.
Uses the same train/val/test split as the training to properly evaluate on held-out data.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
from tqdm import tqdm
import random
from torch.utils.data import random_split
from cine_2d_time_segmentation import Cine2DTimeDataset, Cine2DTimeUNet

# Set random seeds for reproducibility (SAME AS TRAINING)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_trained_model(model_path, temporal_context=5):
    """Load the trained 2D+Time CINE model."""
    model = Cine2DTimeUNet(in_channels=temporal_context + 1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def create_proper_splits(cine_image_dir, cine_label_dir, temporal_context=5):
    """Create the same train/val/test splits as used in training."""
    # Create full dataset
    cine_dataset = Cine2DTimeDataset(cine_image_dir, cine_label_dir, temporal_context=temporal_context)
    
    if len(cine_dataset) == 0:
        print("No valid temporal sequences found!")
        return None, None, None
    
    print(f"Total 2D+Time samples: {len(cine_dataset)}")
    
    # Split dataset (SAME AS TRAINING)
    total_size = len(cine_dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        cine_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def calculate_metrics(prediction, label):
    """Calculate Dice score and IoU."""
    smooth = 1e-6
    prediction_binary = (prediction > 0.5).astype(np.float32)
    
    # Dice score
    intersection = np.sum(prediction_binary * label)
    dice = (2. * intersection + smooth) / (np.sum(prediction_binary) + np.sum(label) + smooth)
    
    # IoU
    union = np.sum(prediction_binary) + np.sum(label) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice, iou

def evaluate_on_holdout_test(model, test_dataset, num_samples=None):
    """Evaluate model on the held-out test set."""
    dice_scores = []
    iou_scores = []
    
    # Use all test samples or specified number
    if num_samples is None:
        sample_indices = list(range(len(test_dataset)))
    else:
        sample_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    print(f"Evaluating on {len(sample_indices)} test samples...")
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Evaluating on held-out test set"):
            input_tensor, label_tensor = test_dataset[idx]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            prediction = model(input_tensor)
            prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
            label = label_tensor.squeeze().numpy()
            
            # Calculate metrics
            dice, iou = calculate_metrics(prediction, label)
            dice_scores.append(dice)
            iou_scores.append(iou)
    
    return dice_scores, iou_scores

def visualize_test_predictions(model, test_dataset, num_samples=6):
    """Visualize predictions on held-out test samples."""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    # Get random test samples
    sample_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample from test set
            input_tensor, label_tensor = test_dataset[idx]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            prediction = model(input_tensor)
            prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
            
            # Get target image (last channel of input)
            target_image = input_tensor.squeeze()[-1].cpu().numpy()
            label = label_tensor.squeeze().numpy()
            
            # Threshold prediction
            prediction_thresholded = (prediction > 0.5).astype(np.float32)
            
            # Plot
            axes[i, 0].imshow(target_image, cmap='gray')
            axes[i, 0].set_title(f'Test Sample {i+1}: Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(label, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(prediction, cmap='hot', alpha=0.7)
            axes[i, 2].imshow(target_image, cmap='gray', alpha=0.3)
            axes[i, 2].set_title('Prediction (Probability)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(prediction_thresholded, cmap='hot', alpha=0.7)
            axes[i, 3].imshow(target_image, cmap='gray', alpha=0.3)
            axes[i, 3].set_title('Prediction (Thresholded)')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('2d_time_cine_holdout_test_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_holdout_performance_metrics(dice_scores, iou_scores):
    """Plot performance metrics for held-out test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dice scores
    axes[0].hist(dice_scores, bins=20, alpha=0.7, color='blue')
    axes[0].axvline(np.mean(dice_scores), color='red', linestyle='--', label=f'Mean: {np.mean(dice_scores):.3f}')
    axes[0].set_title('Distribution of Dice Scores (Held-out Test Set)')
    axes[0].set_xlabel('Dice Score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # IoU scores
    axes[1].hist(iou_scores, bins=20, alpha=0.7, color='green')
    axes[1].axvline(np.mean(iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(iou_scores):.3f}')
    axes[1].set_title('Distribution of IoU Scores (Held-out Test Set)')
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('2d_time_cine_holdout_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== HELD-OUT TEST SET PERFORMANCE ===")
    print(f"Dice Score - Mean: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU Score - Mean: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Number of test samples: {len(dice_scores)}")

def main():
    """Main evaluation function using proper held-out test set."""
    # Paths
    model_path = 'best_2d_time_cine_segmentation_model.pth'
    cine_image_dir = './cardiac_segmentation/cine_classification/images/'
    cine_label_dir = './cardiac_segmentation/cine_classification/labels/'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    # Check if data directories exist
    if not os.path.exists(cine_image_dir) or not os.path.exists(cine_label_dir):
        print("Error: CINE data directories not found!")
        return
    
    # Load trained model
    print("Loading trained model...")
    temporal_context = 5
    model = load_trained_model(model_path, temporal_context)
    print("Model loaded successfully!")
    
    # Create proper splits (same as training)
    print("Creating proper train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_proper_splits(
        cine_image_dir, cine_label_dir, temporal_context
    )
    
    if test_dataset is None:
        print("Failed to create test dataset!")
        return
    
    # Evaluate on held-out test set
    print("Evaluating on held-out test set...")
    dice_scores, iou_scores = evaluate_on_holdout_test(model, test_dataset)
    
    # Plot performance metrics
    print("Plotting performance metrics...")
    plot_holdout_performance_metrics(dice_scores, iou_scores)
    
    # Visualize some test predictions
    print("Generating test prediction visualizations...")
    visualize_test_predictions(model, test_dataset, num_samples=6)
    
    print("\n=== EVALUATION COMPLETED ===")
    print("This evaluation used the proper held-out test set that was never seen during training!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 