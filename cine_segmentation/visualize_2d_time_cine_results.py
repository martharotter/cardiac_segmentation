#!/usr/bin/env python3
"""
Visualization script for 2D+Time CINE segmentation results.
Loads the best trained model and shows predictions on test data.
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
from cine_2d_time_segmentation import Cine2DTimeDataset, Cine2DTimeUNet

# Set random seeds for reproducibility
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

def visualize_temporal_predictions(model, dataset, num_samples=6):
    """Visualize temporal predictions on random samples."""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    # Get random samples
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample
            input_tensor, label_tensor = dataset[idx]
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            prediction = model(input_tensor)
            prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
            
            # Get target image (last channel of input)
            target_image = input_tensor.squeeze()[-1].cpu().numpy()  # Last channel is target
            label = label_tensor.squeeze().numpy()
            
            # Threshold prediction
            prediction_thresholded = (prediction > 0.5).astype(np.float32)
            
            # Plot
            axes[i, 0].imshow(target_image, cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1}: Input Image')
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
    plt.savefig('2d_time_cine_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_temporal_sequence(model, dataset, sequence_idx=0, num_frames=8):
    """Visualize predictions across a temporal sequence."""
    # Get a temporal sequence
    sequence = dataset.temporal_sequences[sequence_idx]
    
    fig, axes = plt.subplots(3, num_frames, figsize=(4*num_frames, 12))
    
    with torch.no_grad():
        for t in range(num_frames):
            if t + 5 < len(sequence):  # Need temporal_context + 1 frames
                # Get sample at time t
                sample_idx = t
                input_tensor, label_tensor = dataset[sample_idx]
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                # Get prediction
                prediction = model(input_tensor)
                prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
                
                # Get target image
                target_image = input_tensor.squeeze()[-1].cpu().numpy()
                label = label_tensor.squeeze().numpy()
                
                # Threshold prediction
                prediction_thresholded = (prediction > 0.5).astype(np.float32)
                
                # Plot
                axes[0, t].imshow(target_image, cmap='gray')
                axes[0, t].set_title(f'Frame {t}')
                axes[0, t].axis('off')
                
                axes[1, t].imshow(label, cmap='gray')
                axes[1, t].set_title('Ground Truth')
                axes[1, t].axis('off')
                
                axes[2, t].imshow(prediction_thresholded, cmap='hot', alpha=0.7)
                axes[2, t].imshow(target_image, cmap='gray', alpha=0.3)
                axes[2, t].set_title('Prediction')
                axes[2, t].axis('off')
    
    plt.tight_layout()
    plt.savefig('2d_time_cine_temporal_sequence.png', dpi=300, bbox_inches='tight')
    plt.show()

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

def evaluate_model_performance(model, dataset, num_samples=100):
    """Evaluate model performance on random samples."""
    dice_scores = []
    iou_scores = []
    
    # Get random samples
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Evaluating"):
            input_tensor, label_tensor = dataset[idx]
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

def plot_performance_metrics(dice_scores, iou_scores):
    """Plot performance metrics distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dice scores
    axes[0].hist(dice_scores, bins=20, alpha=0.7, color='blue')
    axes[0].axvline(np.mean(dice_scores), color='red', linestyle='--', label=f'Mean: {np.mean(dice_scores):.3f}')
    axes[0].set_title('Distribution of Dice Scores')
    axes[0].set_xlabel('Dice Score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # IoU scores
    axes[1].hist(iou_scores, bins=20, alpha=0.7, color='green')
    axes[1].axvline(np.mean(iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(iou_scores):.3f}')
    axes[1].set_title('Distribution of IoU Scores')
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('2d_time_cine_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance Summary:")
    print(f"Dice Score - Mean: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU Score - Mean: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")

def main():
    """Main visualization function."""
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
    
    # Create dataset
    print("Creating dataset...")
    dataset = Cine2DTimeDataset(cine_image_dir, cine_label_dir, temporal_context=temporal_context)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Visualize random predictions
    print("Generating prediction visualizations...")
    visualize_temporal_predictions(model, dataset, num_samples=6)
    
    # Visualize temporal sequence
    print("Generating temporal sequence visualization...")
    if len(dataset.temporal_sequences) > 0:
        visualize_temporal_sequence(model, dataset, sequence_idx=0, num_frames=8)
    
    # Evaluate performance
    print("Evaluating model performance...")
    dice_scores, iou_scores = evaluate_model_performance(model, dataset, num_samples=100)
    
    # Plot performance metrics
    print("Plotting performance metrics...")
    plot_performance_metrics(dice_scores, iou_scores)
    
    print("Visualization completed! Check the generated PNG files.")

if __name__ == "__main__":
    main() 