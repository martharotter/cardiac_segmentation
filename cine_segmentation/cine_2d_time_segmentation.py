#!/usr/bin/env python3
"""
2D+Time UNet for CINE cardiac segmentation.
This implementation uses 2D convolutions with temporal context.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random
import torchvision.transforms as transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Cine2DTimeDataset(Dataset):
    def __init__(self, cine_image_dir, cine_label_dir, temporal_context=5, transform=None, filter_empty_labels=True):
        self.cine_image_dir = cine_image_dir
        self.cine_label_dir = cine_label_dir
        self.temporal_context = temporal_context
        self.transform = transform
        self.filter_empty_labels = filter_empty_labels
        
        # Group files by patient and slice
        self.temporal_sequences = self._group_temporal_sequences()
        
        print(f"Found {len(self.temporal_sequences)} temporal sequences")
        print(f"Temporal context: {temporal_context}")
    
    def _group_temporal_sequences(self):
        """Group files by patient and slice to create temporal sequences."""
        # Get all image files
        image_files = glob.glob(os.path.join(self.cine_image_dir, '*.nii.gz'))
        
        # Group by patient and slice
        sequences = {}
        for img_file in image_files:
            base_name = os.path.basename(img_file).replace('.nii.gz', '')
            
            # Parse filename: 001_SA_CINE_slice000_time000
            parts = base_name.split('_')
            if len(parts) >= 4:
                patient_id = parts[0]
                slice_id = parts[2] + '_' + parts[3]  # slice000
                time_id = parts[4]  # time000
                
                key = f"{patient_id}_{slice_id}"
                if key not in sequences:
                    sequences[key] = []
                
                sequences[key].append((img_file, base_name, time_id))
        
        # Sort each sequence by time and filter valid sequences
        valid_sequences = []
        for key, files in sequences.items():
            # Sort by time
            files.sort(key=lambda x: x[2])
            
            # Check if we have enough temporal frames
            if len(files) >= self.temporal_context + 1:  # +1 for target frame
                # Check if all corresponding labels exist and are non-empty
                valid_files = []
                for img_file, base_name, time_id in files:
                    label_file = os.path.join(self.cine_label_dir, base_name + '.nii.gz')
                    
                    if os.path.exists(label_file):
                        if self.filter_empty_labels:
                            try:
                                label_data = nib.load(label_file).get_fdata()
                                if np.sum(label_data) > 0:
                                    valid_files.append((img_file, label_file, base_name, time_id))
                            except:
                                continue
                        else:
                            valid_files.append((img_file, label_file, base_name, time_id))
                
                if len(valid_files) >= self.temporal_context + 1:
                    valid_sequences.append(valid_files)
        
        return valid_sequences
    
    def __len__(self):
        # Each sequence can provide multiple samples
        total_samples = 0
        for sequence in self.temporal_sequences:
            total_samples += len(sequence) - self.temporal_context
        return total_samples
    
    def __getitem__(self, idx):
        # Find which sequence and position this index corresponds to
        sample_count = 0
        for seq_idx, sequence in enumerate(self.temporal_sequences):
            if sample_count + len(sequence) - self.temporal_context > idx:
                # This sequence contains our sample
                local_idx = idx - sample_count
                break
            sample_count += len(sequence) - self.temporal_context
        
        # Get temporal context frames
        context_frames = sequence[local_idx:local_idx + self.temporal_context]
        target_frame = sequence[local_idx + self.temporal_context]
        
        # Load context frames
        context_images = []
        for img_file, label_file, base_name, time_id in context_frames:
            img_data = nib.load(img_file).get_fdata()
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
            context_images.append(img_data)
        
        # Load target frame
        target_img_file, target_label_file, target_base_name, target_time_id = target_frame
        target_img = nib.load(target_img_file).get_fdata()
        target_label = nib.load(target_label_file).get_fdata()
        
        # Normalize target image
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
        target_label = (target_label > 0.5).astype(np.float32)
        
        # Resize to fixed dimensions
        target_size = (256, 256)
        
        # Resize context images
        context_images_resized = []
        for img in context_images:
            img_resized = torch.nn.functional.interpolate(
                torch.FloatTensor(img).unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0).numpy()
            context_images_resized.append(img_resized)
        
        # Resize target image and label
        target_img_resized = torch.nn.functional.interpolate(
            torch.FloatTensor(target_img).unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()
        
        target_label_resized = torch.nn.functional.interpolate(
            torch.FloatTensor(target_label).unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='nearest'
        ).squeeze(0).squeeze(0).numpy()
        
        # Stack context frames as channels
        context_tensor = torch.FloatTensor(np.stack(context_images_resized, axis=0))  # (T, H, W)
        target_tensor = torch.FloatTensor(target_img_resized).unsqueeze(0)  # (1, H, W)
        label_tensor = torch.FloatTensor(target_label_resized).unsqueeze(0)  # (1, H, W)
        
        # Concatenate context and target as input channels
        input_tensor = torch.cat([context_tensor, target_tensor], dim=0)  # (T+1, H, W)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            label_tensor = self.transform(label_tensor)
        
        return input_tensor, label_tensor

class Cine2DTimeUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):  # 5 context + 1 target = 6 channels
        super(Cine2DTimeUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
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
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        
        # Final output
        out = self.final(dec1)
        out = self.sigmoid(out)
        
        return out

def calculate_dice_score(y_true, y_pred):
    """Calculate Dice score."""
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return dice

def cine_dice_loss(pred, target):
    """Dice loss for CINE segmentation."""
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice

def train_2d_time_cine_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    """Train the 2D+Time CINE model."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_2d_time_cine_segmentation_model.pth')
            print(f'Model saved with validation loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_2d_time_cine_model(model, test_loader):
    """Evaluate the 2D+Time CINE model."""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating 2D+Time Model'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            
            # Calculate Dice score for each sample
            for i in range(images.size(0)):
                pred = predictions[i].cpu().numpy().flatten()
                label = labels[i].cpu().numpy().flatten()
                dice = calculate_dice_score(label, pred)
                dice_scores.append(dice)
    
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f'2D+Time Test Results - Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}')
    
    return dice_scores

def main():
    """Main training function."""
    # CINE-specific paths
    cine_image_dir = './cardiac_segmentation/cine_classification/images/'
    cine_label_dir = './cardiac_segmentation/cine_classification/labels/'
    
    # Check if directories exist
    if not os.path.exists(cine_image_dir) or not os.path.exists(cine_label_dir):
        print("Error: CINE directories not found!")
        return
    
    # Create 2D+Time dataset
    temporal_context = 5  # Number of previous frames to use as context
    cine_dataset = Cine2DTimeDataset(cine_image_dir, cine_label_dir, temporal_context=temporal_context)
    
    if len(cine_dataset) == 0:
        print("No valid temporal sequences found!")
        return
    
    print(f"Total 2D+Time samples: {len(cine_dataset)}")
    
    # Split dataset
    total_size = len(cine_dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        cine_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize 2D+Time model
    model = Cine2DTimeUNet(in_channels=temporal_context + 1, out_channels=1).to(device)
    
    # Loss function and optimizer
    criterion = cine_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting 2D+Time CINE segmentation training...")
    train_losses, val_losses = train_2d_time_cine_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('2D+Time CINE Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Evaluate on test set
    dice_scores = evaluate_2d_time_cine_model(model, test_loader)
    
    plt.subplot(1, 2, 2)
    plt.hist(dice_scores, bins=20, alpha=0.7)
    plt.title('Distribution of 2D+Time Dice Scores')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('2d_time_cine_training_results.png')
    plt.show()
    
    print("2D+Time CINE segmentation training completed! Model saved as 'best_2d_time_cine_segmentation_model.pth'")

if __name__ == "__main__":
    main() 