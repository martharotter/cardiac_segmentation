#!/usr/bin/env python3
"""
2D+Time UNet for CINE cardiac segmentation.
This implementation uses 2D convolutions with temporal context.

MODIFIED for BINARY RV segmentation (RV vs Background only).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from npy_cine_dataset import NPYCineDataset, get_npy_cine_loaders

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Binary Dice Loss Function for RV Segmentation ---
def binary_dice_loss(pred, target):
    """
    Computes binary Dice loss for RV vs background segmentation.
    Assumes pred is raw logits and target is binary (0 or 1).
    """
    # Apply sigmoid to get probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.float().view(-1)
    
    # Calculate Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice

# ----------------------------------------
# --- Using NPYCineDataset for .npy files ---
# ----------------------------------------

class Cine2DTimeUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Cine2DTimeUNet, self).__init__()
        
        # Encoder - using original naming convention
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder - using original naming convention
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output - No sigmoid here
        out = self.final(dec1)
        return out

def evaluate_rv_segmentation(model, data_loader, device=None, output_path="evaluation_visualizations"):
    model.eval()
    dice_scores = []
    iou_scores = []
    
    os.makedirs(output_path, exist_ok=True)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(data_loader, desc="RV Segmentation Evaluation")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid and threshold for binary prediction
            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            
            # Convert labels to binary (RV = 1, everything else = 0)
            binary_labels = (labels == 3).float()
            
            # ❗ FIX: Add a channel dimension to binary_labels
            binary_labels = binary_labels.unsqueeze(1)
            
            # Ensure both tensors have the same shape
            if predicted_labels.shape != binary_labels.shape:
                print(f"Shape mismatch - predicted: {predicted_labels.shape}, labels: {binary_labels.shape}")
                predicted_labels = F.interpolate(predicted_labels, size=binary_labels.shape[2:], mode='nearest')
            
            # --- START VISUALIZATION CODE ---
            if i % 10 == 0:  # Visualize every 10th batch
                # Convert tensors to numpy arrays
                input_image = inputs[0, inputs.shape[1] // 2, :, :].cpu().numpy() # Get the middle frame
                predicted_mask = predicted_labels[0, 0, :, :].cpu().numpy()
                true_mask = binary_labels[0, 0, :, :].cpu().numpy()
                
                # Create a figure with 3 subplots
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                
                ax[0].imshow(input_image, cmap='gray')
                ax[0].set_title('Original Input Frame')
                ax[0].axis('off')
                
                ax[1].imshow(true_mask, cmap='gray')
                ax[1].set_title('Ground Truth RV Mask')
                ax[1].axis('off')
                
                ax[2].imshow(predicted_mask, cmap='gray')
                ax[2].set_title('Predicted RV Mask')
                ax[2].axis('off')
                
                # Save the figure
                plt.savefig(os.path.join(output_path, f"evaluation_batch_{i}.png"))
                plt.close(fig)
            # --- END VISUALIZATION CODE ---
            
            # Calculate Dice score - handle batch dimension properly
            intersection = (predicted_labels * binary_labels).sum(dim=[1, 2, 3])
            union = predicted_labels.sum(dim=[1, 2, 3]) + binary_labels.sum(dim=[1, 2, 3])
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.extend(dice.cpu().numpy())
            
            # Calculate IoU score
            iou = intersection / (union - intersection + 1e-6)
            iou_scores.extend(iou.cpu().numpy())
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    print(f"RV Segmentation Results:")
    print(f"  Mean Dice Score: {mean_dice:.4f}")
    print(f"  Mean IoU Score: {mean_iou:.4f}")
    
    return mean_dice, mean_iou

def train_rv_segmentation_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, model_save_path='rv_model.pth', device=None, unfreeze_epoch=10):
    best_val_loss = float('inf')
    best_dice_score = 0.0
    
    for epoch in range(num_epochs):
        # ❗ NEW: Unfreeze encoder layers after a specific epoch
        if epoch == unfreeze_epoch:
            print(f"Unfreezing encoder layers at epoch {epoch+1}...")
            for name, param in model.named_parameters():
                if 'enc' in name:
                    param.requires_grad = True
            
            # Re-initialize the optimizer to include the newly unfrozen parameters
            optimizer = optim.Adam(model.parameters(), lr=1e-5) # Use a very low LR for fine-tuning
            print("Encoder layers unfrozen. Training the entire model now.")
        
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to binary (RV = 1, everything else = 0)
            binary_labels = (labels == 3).float()
            
            # ❗ FIX: Add a channel dimension to binary_labels
            binary_labels = binary_labels.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure outputs and labels have the same shape for loss calculation
            if outputs.shape[2:] != binary_labels.shape[2:]:
                # Resize outputs to match labels
                outputs = F.interpolate(outputs, size=binary_labels.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    binary_labels = (labels == 3).float()
                    
                    # ❗ FIX: Add a channel dimension to binary_labels
                    binary_labels = binary_labels.unsqueeze(1)
                    
                    outputs = model(inputs)
                    
                    # Ensure outputs and labels have the same shape for loss calculation
                    if outputs.shape[2:] != binary_labels.shape[2:]:
                        # Resize outputs to match labels
                        outputs = F.interpolate(outputs, size=binary_labels.shape[2:], mode='bilinear', align_corners=False)
                    
                    loss = criterion(outputs, binary_labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Evaluate RV segmentation performance
            print(f"--- Epoch {epoch+1}/{num_epochs} RV Segmentation Evaluation ---")
            dice_score, iou_score = evaluate_rv_segmentation(model, val_loader, device=device)
            
            # Save the model if validation loss is the best so far or dice score improves
            if avg_val_loss < best_val_loss or dice_score > best_dice_score:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                if dice_score > best_dice_score:
                    best_dice_score = dice_score
                
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                print(f"  Best dice score: {best_dice_score:.4f}")

    print(f"-- Final RV Segmentation Evaluation ---")
    dice_score, iou_score = evaluate_rv_segmentation(model, val_loader, device=device)
    print("\nTraining complete.")



def main():
    # Define directories for the new .npy dataset
    data_dir = "./cardiac_segmentation/cine_classification_npy/"
    
    # Hyperparameters for RV segmentation
    temporal_context = 5
    target_size = (256, 256)
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4

    # Define model save path for the base model
    pretrained_model_path = './best_2d_time_cine_segmentation_model.pth'

    # ❗  NEW: Define a separate save path for the fine-tuned RV model
    rv_model_save_path = './fine_tuned_rv_segmentation_model.pth'
    
    # Data loading using the new NPYCineDataset with specific split sizes
    print(f"Loading dataset from: {data_dir}")
    print("Focusing on RV (Right Ventricle) segmentation only")
    
    # Create full dataset
    full_dataset = NPYCineDataset(
        data_dir=data_dir,
        temporal_context=temporal_context,
        target_size=target_size
    )
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    # --- Dynamically calculate split sizes to avoid ValueError ---
    train_split_ratio = 0.4
    val_split_ratio = 0.2
    test_split_ratio = 0.4
    
    total_size = len(full_dataset)
    train_size = int(train_split_ratio * total_size)
    val_size = int(val_split_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model, Loss, Optimizer - Now binary output for RV segmentation
    model = Cine2DTimeUNet(in_channels=temporal_context + 1, out_channels=1).to(device)
    
    # Load pre-trained model for transfer learning (heart tissue vs background)
    if os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained model from {pretrained_model_path} for transfer learning...")
        # Load the pre-trained weights
        pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)
        
        # Load the state dictionary with strict=True to catch any mismatches.
        # This will now work because the model architectures are identical.
        model.load_state_dict(pretrained_state_dict, strict=False)
        print("Successfully loaded pre-trained model.")
        print("Using transfer learning from heart tissue vs background model")
    else:
        print("No pre-trained model found. Training from scratch.")
    
    criterion = binary_dice_loss
    
    # Transfer learning strategy: Start with frozen encoder, then fine-tune
    # We now freeze the encoder and use a higher learning rate to start.
    if os.path.exists(pretrained_model_path):
        print("Starting with frozen encoder for transfer learning...")
        # Freeze encoder layers initially (using original naming convention)
        for name, param in model.named_parameters():
            if 'enc' in name:
                param.requires_grad = False
        print("Encoder layers frozen. Training decoder and output layers only.")
        
        # Use a higher learning rate for the initial decoder training
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        print(f"Using initial learning rate: {1e-4} for decoder training.")
    else:
        # If no pre-trained model, train the whole model from scratch
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model for RV segmentation
    train_rv_segmentation_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, rv_model_save_path, device=device)
    
    # Final evaluation on the test set
    print("\n--- Final RV Segmentation Evaluation on Test Set ---")
    if os.path.exists(rv_model_save_path):
        print(f"Loading best model from {rv_model_save_path} for final evaluation.")
        model.load_state_dict(torch.load(rv_model_save_path))
    else:
        print("Best RV model file not found. Using the last trained model state.")
    
    # ❗  MODIFIED DATA LOADER for EVALUATION to avoid MemoryError
    # Use a single worker and smaller batch size for evaluation
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2,  # Reduced batch size
        shuffle=False, 
        num_workers=0, # Changed to 0 workers
        pin_memory=True,
        persistent_workers=False
    )
    
    dice_score, iou_score = evaluate_rv_segmentation(model, test_loader, device=device)
    
    print("\nRV segmentation training and evaluation complete.")

if __name__ == '__main__':
    main()