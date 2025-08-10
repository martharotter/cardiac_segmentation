import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import csv
from datetime import datetime
import winsound
import time

# Set paths
IMG_DIR = 'C:/Users/willi/Documents/School/Medical Scanning/cardiac_segmentation/out/images/'
LBL_DIR = 'C:/Users/willi/Documents/School/Medical Scanning/cardiac_segmentation/out/labels/'

# List and sort files for reproducibility
img_files = sorted(glob.glob(os.path.join(IMG_DIR, '*.nii.gz')))
lbl_files = sorted(glob.glob(os.path.join(LBL_DIR, '*.nii.gz')))

assert len(img_files) == len(lbl_files), "Mismatch between images and labels!"

# Pair images and labels by filename
pairs = list(zip(img_files, lbl_files))

# Split: 160 train, 40 val, 160 test
train_pairs = pairs[:160]
val_pairs = pairs[160:200]
test_pairs = pairs[200:]



# Custom Dataset
class MRIDataset(Dataset):
    def __init__(self, pairs, target_size=(256, 256), transform=None):
        self.pairs = pairs
        self.transform = transform
        self.target_size = target_size
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int64)
        # Normalize image
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        lbl = np.expand_dims(lbl, axis=0)  # Add channel for transform
        img = torch.tensor(img, dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.long)
        # Resize both to target size
        img = TF.resize(img, self.target_size)
        lbl = TF.resize(lbl, self.target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return img, lbl

# U-Net Model (simple 2D version)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.dconv_down1 = DoubleConv(1, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 128)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dconv_up2 = DoubleConv(128, 64)   # 64 (skip) + 64 (up) = 128 -> 64
        self.dconv_up1 = DoubleConv(64, 32)    # 32 (skip) + 32 (up) = 64 -> 32
        self.conv_last = nn.Conv2d(32, n_classes, 1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dconv_down3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

# U-Net++ Model (Nested U-Net)
class UNetPlusPlus(nn.Module):
    def __init__(self, n_classes, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.conv0_0 = DoubleConv(1, 32)
        self.conv1_0 = DoubleConv(32, 64)
        self.conv2_0 = DoubleConv(64, 128)
        self.conv3_0 = DoubleConv(128, 256)
        
        # Decoder with nested connections
        self.conv0_1 = DoubleConv(32 + 64, 32)
        self.conv1_1 = DoubleConv(64 + 128, 64)
        self.conv2_1 = DoubleConv(128 + 256, 128)
        
        self.conv0_2 = DoubleConv(32 * 2 + 64, 32)
        self.conv1_2 = DoubleConv(64 * 2 + 128, 64)
        
        self.conv0_3 = DoubleConv(32 * 3 + 64, 32)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final conv
        self.final1 = nn.Conv2d(32, n_classes, kernel_size=1)
        self.final2 = nn.Conv2d(32, n_classes, kernel_size=1)
        self.final3 = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        
        # Decoder with nested connections
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]
        else:
            output = self.final3(x0_3)
            return output
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# MAnet Model (Multi-scale Attention Network)
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        return x * attention

class MAnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(1, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 256)
        
        # Attention modules
        self.attention1 = AttentionModule(32)
        self.attention2 = AttentionModule(64)
        self.attention3 = AttentionModule(128)
        self.attention4 = AttentionModule(256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        self.conv_up3 = DoubleConv(256, 128)  # 128 (skip) + 128 (up) = 256
        self.conv_up2 = DoubleConv(128, 64)   # 64 (skip) + 64 (up) = 128
        self.conv_up1 = DoubleConv(64, 32)    # 32 (skip) + 32 (up) = 64
        
        self.final = nn.Conv2d(32, n_classes, 1)
        
    def forward(self, x):
        # Encoder with attention
        conv1 = self.attention1(self.conv1(x))
        conv2 = self.attention2(self.conv2(self.pool(conv1)))
        conv3 = self.attention3(self.conv3(self.pool(conv2)))
        conv4 = self.attention4(self.conv4(self.pool(conv3)))
        
        # Decoder with skip connections
        up3 = self.up3(conv4)
        up3 = torch.cat([up3, conv3], dim=1)
        up3 = self.conv_up3(up3)
        
        up2 = self.up2(up3)
        up2 = torch.cat([up2, conv2], dim=1)
        up2 = self.conv_up2(up2)
        
        up1 = self.up1(up2)
        up1 = torch.cat([up1, conv1], dim=1)
        up1 = self.conv_up1(up1)
        
        return self.final(up1)
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# Linknet Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = torch.relu(out)
        
        return out

class Linknet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder1 = ResidualBlock(64, 64)
        self.encoder2 = ResidualBlock(64, 128)
        self.encoder3 = ResidualBlock(128, 256)
        self.encoder4 = ResidualBlock(256, 512)
        
        # Decoder
        self.decoder4 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        
        # Final conv
        self.final = nn.Conv2d(32, n_classes, 1)
        
    def forward(self, x):
        # Encoder
        x = self.initial(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Decoder
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        return self.final(d1)
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# FPN Model (Feature Pyramid Network)
class FPN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder (ResNet-like backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        self.stage1 = self._make_layer(64, 64, 2)   # 1/2
        self.stage2 = self._make_layer(64, 128, 2)  # 1/4
        self.stage3 = self._make_layer(128, 256, 2) # 1/8
        self.stage4 = self._make_layer(256, 512, 2) # 1/16
        
        # FPN lateral connections
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(128, 256, 1)
        self.lateral1 = nn.Conv2d(64, 256, 1)
        
        # FPN output convolutions
        self.fpn4 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn3 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn2 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn1 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, 1)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        c1 = self.stage1(x)
        c2 = self.stage2(self.pool(c1))
        c3 = self.stage3(self.pool(c2))
        c4 = self.stage4(self.pool(c3))
        
        # FPN top-down pathway
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + self.upsample(p4)
        p2 = self.lateral2(c2) + self.upsample(p3)
        p1 = self.lateral1(c1) + self.upsample(p2)
        
        # FPN output convolutions
        p4 = self.fpn4(p4)
        p3 = self.fpn3(p3)
        p2 = self.fpn2(p2)
        p1 = self.fpn1(p1)
        
        # Use the finest level (p1) for segmentation
        # Upsample to original resolution
        p1 = self.upsample(p1)  # 1/2 -> 1/1
        
        return self.seg_head(p1)
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# PSPNet Model (Pyramid Scene Parsing Network)
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes):
        super().__init__()
        self.bin_sizes = bin_sizes
        self.pools = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        for bin_size in bin_sizes:
            self.pools.append(nn.AdaptiveAvgPool2d(bin_size))
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x):
        size = x.size()
        out = [x]
        
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = nn.functional.interpolate(conv_out, size=size[2:], mode='bilinear', align_corners=True)
            out.append(upsampled)
        
        return torch.cat(out, dim=1)

class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder (ResNet-like backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        self.stage1 = self._make_layer(64, 64, 2)   # 1/2
        self.stage2 = self._make_layer(64, 128, 2)  # 1/4
        self.stage3 = self._make_layer(128, 256, 2) # 1/8
        self.stage4 = self._make_layer(256, 512, 2) # 1/16
        
        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(512, 128, [1, 2, 3, 6])
        
        # Final convolutions
        self.final_conv = nn.Sequential(
            nn.Conv2d(512 + 4*128, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_classes, 1)
        )
        
        # Upsampling to original resolution (256x256)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(self.pool(x))
        x = self.stage3(self.pool(x))
        x = self.stage4(self.pool(x))
        
        # Pyramid Pooling Module
        ppm_out = self.ppm(x)
        
        # Final convolution
        out = self.final_conv(ppm_out)
        
        # Upsample to original resolution
        out = self.upsample(out)
        
        return out
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# PAN Model (Pyramid Attention Network)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class PyramidAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1)
        
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        
        self.final_conv = nn.Conv2d(4 * out_channels, out_channels, 1)
        
    def forward(self, x):
        size = x.size()[2:]
        
        # Multi-scale pooling
        feat1 = self.pool1(x)
        feat2 = self.pool2(x)
        feat3 = self.pool3(x)
        feat4 = self.pool4(x)
        
        # Convolutions
        feat1 = self.conv1(feat1)
        feat2 = self.conv2(feat2)
        feat3 = self.conv3(feat3)
        feat4 = self.conv4(feat4)
        
        # Upsample to original size
        feat1 = nn.functional.interpolate(feat1, size=size, mode='bilinear', align_corners=True)
        feat2 = nn.functional.interpolate(feat2, size=size, mode='bilinear', align_corners=True)
        feat3 = nn.functional.interpolate(feat3, size=size, mode='bilinear', align_corners=True)
        feat4 = nn.functional.interpolate(feat4, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate and final convolution
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.final_conv(out)
        
        return out

class PAN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder (ResNet-like backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        self.stage1 = self._make_layer(64, 64, 2)   # 1/2
        self.stage2 = self._make_layer(64, 128, 2)  # 1/4
        self.stage3 = self._make_layer(128, 256, 2) # 1/8
        self.stage4 = self._make_layer(256, 512, 2) # 1/16
        
        # Pyramid Attention Module
        self.pam = PyramidAttention(512, 128)
        
        # Spatial Attention Module
        self.sam = SpatialAttention()
        
        # Final convolutions
        self.final_conv = nn.Sequential(
            nn.Conv2d(512 + 128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
        
        # Upsampling to original resolution (256x256)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(self.pool(x))
        x = self.stage3(self.pool(x))
        x = self.stage4(self.pool(x))
        
        # Pyramid Attention Module
        pam_out = self.pam(x)
        
        # Spatial Attention Module
        sam_out = self.sam(x)
        
        # Combine features
        combined = torch.cat([sam_out, pam_out], dim=1)
        
        # Final convolution
        out = self.final_conv(combined)
        
        # Upsample to original resolution
        out = self.upsample(out)
        
        return out
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# DeepLabV3 Model
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
    
    def forward(self, x):
        size = x.size()[2:]
        
        # Apply different convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        
        # Global average pooling
        global_feat = self.global_avg_pool(x)
        global_feat = nn.functional.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate all features
        out = torch.cat([conv1, conv2, conv3, conv4, global_feat], dim=1)
        out = self.final_conv(out)
        
        return out

class DeepLabV3(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder (ResNet-like backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        self.stage1 = self._make_layer(64, 64, 2)   # 1/2
        self.stage2 = self._make_layer(64, 128, 2)  # 1/4
        self.stage3 = self._make_layer(128, 256, 2) # 1/8
        self.stage4 = self._make_layer(256, 512, 2) # 1/16
        
        # ASPP module
        self.aspp = ASPP(512, 256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
        
        # Upsampling to original resolution (256x256)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(self.pool(x))
        x = self.stage3(self.pool(x))
        x = self.stage4(self.pool(x))
        
        # ASPP module
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to original resolution
        x = self.upsample(x)
        
        return x
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# DeepLabV3+ Model
class Decoder(nn.Module):
    def __init__(self, low_level_channels, aspp_channels, decoder_channels):
        super().__init__()
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, aspp_features, low_level_features):
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        
        # Upsample ASPP features to match low-level feature size
        aspp_features = nn.functional.interpolate(
            aspp_features, size=low_level_features.size()[2:], 
            mode='bilinear', align_corners=True
        )
        
        # Concatenate and process
        combined = torch.cat([aspp_features, low_level_features], dim=1)
        output = self.decoder_conv(combined)
        
        return output

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder (ResNet-like backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        self.stage1 = self._make_layer(64, 64, 2)   # 1/2 - save for decoder
        self.stage2 = self._make_layer(64, 128, 2)  # 1/4
        self.stage3 = self._make_layer(128, 256, 2) # 1/8
        self.stage4 = self._make_layer(256, 512, 2) # 1/16
        
        # ASPP module
        self.aspp = ASPP(512, 256)
        
        # Decoder
        self.decoder = Decoder(64, 256, 256)  # 64 from stage1, 256 from ASPP
        
        # Final segmentation head
        self.final_conv = nn.Conv2d(256, n_classes, 1)
        
        # Upsampling to original resolution (256x256)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        low_level_feat = self.stage1(x)  # Save for decoder
        x = self.stage2(self.pool(low_level_feat))
        x = self.stage3(self.pool(x))
        x = self.stage4(self.pool(x))
        
        # ASPP module
        aspp_features = self.aspp(x)
        
        # Decoder
        decoder_features = self.decoder(aspp_features, low_level_feat)
        
        # Final convolution
        output = self.final_conv(decoder_features)
        
        # Upsample to original resolution
        output = self.upsample(output)
        
        return output
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# UPerNet Model (Unified Perceptual Parsing Network)
class PPM(nn.Module):
    """Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels, bin_sizes):
        super().__init__()
        self.bin_sizes = bin_sizes
        self.pools = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        for bin_size in bin_sizes:
            self.pools.append(nn.AdaptiveAvgPool2d(bin_size))
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x):
        size = x.size()[2:]
        out = [x]
        
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = nn.functional.interpolate(conv_out, size=size, mode='bilinear', align_corners=True)
            out.append(upsampled)
        
        return torch.cat(out, dim=1)

class FPNHead(nn.Module):
    """Feature Pyramid Network Head"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UPerNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Encoder (ResNet-like backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        self.stage1 = self._make_layer(64, 64, 2)   # 1/2
        self.stage2 = self._make_layer(64, 128, 2)  # 1/4
        self.stage3 = self._make_layer(128, 256, 2) # 1/8
        self.stage4 = self._make_layer(256, 512, 2) # 1/16
        
        # Pyramid Pooling Module
        self.ppm = PPM(512, 128, [1, 2, 3, 6])
        
        # Simplified FPN - just use the PPM output directly
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(512 + 4*128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(self.pool(x))
        x = self.stage3(self.pool(x))
        x = self.stage4(self.pool(x))
        
        # Pyramid Pooling Module
        ppm_out = self.ppm(x)
        
        # Simplified processing
        features = self.fpn_conv(ppm_out)
        
        # Final segmentation
        out = self.seg_head(features)
        
        # Upsample to original resolution
        out = self.final_upsample(out)
        
        return out
    
    def pool(self, x):
        return nn.MaxPool2d(2)(x)

# Segformer Model (Transformer-based segmentation)
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_channels=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class Segformer(nn.Module):
    def __init__(self, n_classes, img_size=256, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], 
                 depths=[2, 2, 2, 2], patch_size=4):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dims[0])
        
        # Transformer stages
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], drop=0.1, attn_drop=0.1)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                # Downsample to next stage
                self.stages.append(nn.Conv2d(embed_dims[i], embed_dims[i+1], 2, stride=2))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
        
        # Final upsampling
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Transformer stages
        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.ModuleList):
                # Transformer blocks
                for block in stage:
                    x = block(x)
            else:
                # Downsampling
                # Reshape to 2D for convolution
                B, n_patches, embed_dim = x.shape
                h = w = int(n_patches ** 0.5)
                x = x.transpose(1, 2).reshape(B, embed_dim, h, w)
                x = stage(x)  # Downsample
                # Reshape back to sequence
                B, embed_dim, h, w = x.shape
                x = x.flatten(2).transpose(1, 2)
        
        # Reshape to 2D for decoder
        B, n_patches, embed_dim = x.shape
        h = w = int(n_patches ** 0.5)
        x = x.transpose(1, 2).reshape(B, embed_dim, h, w)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to original resolution
        x = self.upsample(x)
        
        return x

# DPT Model (Dense Prediction Transformer)
class DPTHead(nn.Module):
    """DPT Decoder Head"""
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv_layers(x)

class DPT(nn.Module):
    def __init__(self, n_classes, img_size=256, embed_dim=768, num_heads=12, 
                 num_layers=12, patch_size=16, mlp_ratio=4.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=0.1, attn_drop=0.1)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # DPT decoder heads
        self.head = DPTHead(embed_dim, 256)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, 1)
        )
        
        # Upsampling to original resolution
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Reshape to 2D for decoder
        B, num_patches, embed_dim = x.shape
        h = w = int(num_patches ** 0.5)
        x = x.transpose(1, 2).reshape(B, embed_dim, h, w)
        
        # DPT decoder
        x = self.head(x)
        
        # Final segmentation
        x = self.seg_head(x)
        
        # Upsample to original resolution
        x = self.upsample(x)
        
        return x

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        logits = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        intersection = torch.sum(logits * targets_onehot, dims)
        union = torch.sum(logits, dims) + torch.sum(targets_onehot, dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def play_completion_sound():
    """Play a sound to notify that training is complete"""
    try:
        # Play a system sound (Windows)
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        # Also play a beep sound
        winsound.Beep(1000, 500)  # 1000Hz for 500ms
        print("\n🎵 Training completed! Sound notification played.")
    except:
        # Fallback for non-Windows systems or if sound fails
        print("\n🎵 Training completed! (Sound notification not available)")
        # Print some visual indicators
        for _ in range(3):
            print("🔔 DING DING DING - TRAINING FINISHED! 🔔")
            time.sleep(0.5)

def get_loaders(batch_size=8):
    train_ds = MRIDataset(train_pairs)
    val_ds = MRIDataset(val_pairs)
    test_ds = MRIDataset(test_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model_type='unet', learning_rate=0.001, batch_size=8, epochs=50, weight_decay=0.0001, 
                class_weights=None, optimizer_type='adam'):
    """
    Train a segmentation model with specified hyperparameters.
    
    Args:
        model_type: Type of model to train
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        weight_decay: Weight decay for regularization
        class_weights: Class weights for loss function
        optimizer_type: Type of optimizer ('adam' or 'sgd')
    """
    """
    Train a segmentation model.
    
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    n_classes = 4  # 0=background, 1=LV, 2=MYO, 3=RV
    
    # Model selection
    if model_type == 'unet':
        model = UNet(n_classes).to(device)
        print("Using U-Net model")
    elif model_type == 'unetpp':
        model = UNetPlusPlus(n_classes).to(device)
        print("Using U-Net++ model")
    elif model_type == 'manet':
        model = MAnet(n_classes).to(device)
        print("Using MAnet model")
    elif model_type == 'linknet':
        model = Linknet(n_classes).to(device)
        print("Using Linknet model")
    elif model_type == 'fpn':
        model = FPN(n_classes).to(device)
        print("Using FPN model")
    elif model_type == 'pspnet':
        model = PSPNet(n_classes).to(device)
        print("Using PSPNet model")
    elif model_type == 'pan':
        model = PAN(n_classes).to(device)
        print("Using PAN model")
    elif model_type == 'deeplabv3':
        model = DeepLabV3(n_classes).to(device)
        print("Using DeepLabV3 model")
    elif model_type == 'deeplabv3plus':
        model = DeepLabV3Plus(n_classes).to(device)
        print("Using DeepLabV3+ model")
    elif model_type == 'upernet':
        model = UPerNet(n_classes).to(device)
        print("Using UPerNet model")
    elif model_type == 'segformer':
        model = Segformer(n_classes).to(device)
        print("Using Segformer model")
    elif model_type == 'dpt':
        model = DPT(n_classes).to(device)
        print("Using DPT model")
    else:
        model = UNet(n_classes).to(device)
        print("Using U-Net model (default)")
    
    # Create results directory (same directory as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', model_type)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    
    # Create CSV file for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f'training_log_{model_type}_{timestamp}.csv')
    print(f"CSV file will be saved to: {csv_filename}")
    
    # Use weighted CrossEntropyLoss for class imbalance
    # Set up class weights based on hyperparameters
    if class_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights_tensor = torch.tensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Set up optimizer based on hyperparameters
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, val_loader, test_loader = get_loaders()
    best_val_loss = float('inf')
    num_epochs = epochs
    
    # Initialize CSV with headers
    csv_writer = None
    try:
        csvfile = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Test_Loss', 'Mean_Dice'])
        print("✓ CSV headers written successfully")
    except Exception as e:
        print(f"✗ Error initializing CSV: {e}")
        if csvfile:
            csvfile.close()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (imgs, lbls) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Log to CSV
        try:
            if csv_writer:
                csv_writer.writerow([epoch+1, f'{train_loss:.4f}', f'{val_loss:.4f}', '', ''])
                csvfile.flush()  # Ensure data is written to disk
            if epoch % 10 == 0:  # Print every 10 epochs to confirm writing
                print(f"CSV logged epoch {epoch+1}")
        except Exception as e:
            print(f"Error writing to CSV: {e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_{model_type}.pth')
            print("Saved new best model.")
        if epoch == 0 and batch_idx == 0:
            print(f"Model output shape: {outputs.shape}")
            print(f"Model output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            print(f"Target labels unique: {torch.unique(lbls).tolist()}")
    # Test
    model.load_state_dict(torch.load(f'best_{model_type}.pth'))
    model.eval()
    test_loss = 0
    dice_scores = []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Testing"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            test_loss += loss.item() * imgs.size(0)
            
            # Calculate Dice for each class
            preds = torch.softmax(outputs, dim=1)  # Get probabilities
            for c in range(1, 4):  # Skip background (class 0)
                pred_mask = (preds[:, c] > 0.5).float()  # Threshold at 0.5
                target_mask = (lbls == c).float()
                
                intersection = (pred_mask * target_mask).sum()
                union = pred_mask.sum() + target_mask.sum()
                
                dice = (2 * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

    test_loss /= len(test_loader.dataset)
    mean_dice = np.mean(dice_scores)
    print(f"Test Loss: {test_loss:.4f}, Mean Dice: {mean_dice:.4f}")
    
    # Log final test results to CSV and close file
    try:
        if csv_writer:
            csv_writer.writerow(['FINAL', '', '', f'{test_loss:.4f}', f'{mean_dice:.4f}'])
            csvfile.close()
            print(f"Training log saved to: {csv_filename}")
    except Exception as e:
        print(f"Error writing final results to CSV: {e}")
        print(f"Attempted to write to: {csv_filename}")
        if csvfile:
            csvfile.close()
    
    # Play completion sound
    play_completion_sound()

def hyperparameter_tuning(model_type='unet', num_trials=10):
    """
    Perform hyperparameter tuning for a given model type.
    
    Args:
        model_type: Type of model to tune
        num_trials: Number of hyperparameter combinations to try
    """
    import itertools
    import json
    from datetime import datetime
    
    # Define hyperparameter search space
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    batch_sizes = [4, 8, 16]
    epochs_list = [30, 50, 100]
    weight_decays = [0.00001, 0.0001, 0.001]
    optimizers = ['adam', 'sgd']
    class_weight_configs = [
        None,  # No class weights
        [1.0, 5.0, 5.0, 5.0],  # Moderate class weights
        [1.0, 10.0, 10.0, 10.0],  # High class weights
        [1.0, 15.0, 15.0, 15.0]   # Very high class weights
    ]
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        learning_rates, batch_sizes, epochs_list, weight_decays, optimizers, class_weight_configs
    ))
    
    # Randomly sample combinations if we have too many
    if len(all_combinations) > num_trials:
        import random
        random.shuffle(all_combinations)
        all_combinations = all_combinations[:num_trials]
    
    results = []
    best_score = 0
    best_params = None
    
    print(f"Starting hyperparameter tuning for {model_type}")
    print(f"Testing {len(all_combinations)} combinations...")
    
    for i, (lr, bs, epochs, wd, opt, cw) in enumerate(all_combinations):
        print(f"\n--- Trial {i+1}/{len(all_combinations)} ---")
        print(f"Testing: lr={lr}, batch_size={bs}, epochs={epochs}, weight_decay={wd}, optimizer={opt}, class_weights={cw}")
        
        try:
            # Train model with these hyperparameters
            train_model(
                model_type=model_type,
                learning_rate=lr,
                batch_size=bs,
                epochs=epochs,
                weight_decay=wd,
                class_weights=cw,
                optimizer_type=opt
            )
            
            # Read the results from the CSV file
            import glob
            csv_files = glob.glob(f'./results/{model_type}/training_log_{model_type}_*.csv')
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getctime)
                with open(latest_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # Get the final test results
                        final_line = lines[-1]
                        if final_line.startswith('FINAL'):
                            parts = final_line.strip().split(',')
                            if len(parts) >= 5:
                                test_loss = float(parts[3])
                                mean_dice = float(parts[4])
                                
                                result = {
                                    'trial': i+1,
                                    'learning_rate': lr,
                                    'batch_size': bs,
                                    'epochs': epochs,
                                    'weight_decay': wd,
                                    'optimizer': opt,
                                    'class_weights': cw,
                                    'test_loss': test_loss,
                                    'mean_dice': mean_dice
                                }
                                results.append(result)
                                
                                print(f"Results: Test Loss={test_loss:.4f}, Mean Dice={mean_dice:.4f}")
                                
                                # Update best parameters
                                if mean_dice > best_score:
                                    best_score = mean_dice
                                    best_params = result
                                    print(f"*** NEW BEST! Dice Score: {mean_dice:.4f} ***")
            
        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            continue
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'./results/{model_type}/hyperparameter_tuning_{model_type}_{timestamp}.json'
    os.makedirs(f'./results/{model_type}/', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'model_type': model_type,
            'total_trials': len(all_combinations),
            'successful_trials': len(results),
            'results': results,
            'best_parameters': best_params
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING SUMMARY FOR {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Total trials: {len(all_combinations)}")
    print(f"Successful trials: {len(results)}")
    
    if best_params:
        print(f"\nBEST PARAMETERS:")
        print(f"  Learning Rate: {best_params['learning_rate']}")
        print(f"  Batch Size: {best_params['batch_size']}")
        print(f"  Epochs: {best_params['epochs']}")
        print(f"  Weight Decay: {best_params['weight_decay']}")
        print(f"  Optimizer: {best_params['optimizer']}")
        print(f"  Class Weights: {best_params['class_weights']}")
        print(f"  Test Loss: {best_params['test_loss']:.4f}")
        print(f"  Mean Dice Score: {best_params['mean_dice']:.4f}")
    
    # Sort results by Dice score
    results.sort(key=lambda x: x['mean_dice'], reverse=True)
    print(f"\nTOP 5 CONFIGURATIONS:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Dice: {result['mean_dice']:.4f} | "
              f"LR: {result['learning_rate']} | "
              f"BS: {result['batch_size']} | "
              f"Epochs: {result['epochs']} | "
              f"Opt: {result['optimizer']}")
    
    print(f"\nResults saved to: {results_file}")
    return best_params

def visualize_pairs_interactive(dataset, num_samples=10):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    
    fig, (ax_img, ax_lbl) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.15)  # Make room for buttons
    
    current_idx = 0
    
    def update_plot():
        img, lbl = dataset[current_idx]
        
        # Clear previous plots
        ax_img.clear()
        ax_lbl.clear()
        
        # Plot image
        ax_img.imshow(img.squeeze(), cmap='gray')
        ax_img.set_title(f'Sample {current_idx}: Image\n{img_path if "img_path" in locals() else ""}')
        ax_img.axis('off')
        
        # Plot label
        ax_lbl.imshow(lbl, cmap='tab10')
        ax_lbl.set_title(f'Sample {current_idx}: Label\nUnique values: {torch.unique(lbl).tolist()}')
        ax_lbl.axis('off')
        
        plt.draw()
    
    def next_sample(event):
        nonlocal current_idx
        current_idx = (current_idx + 1) % min(num_samples, len(dataset))
        update_plot()
    
    def prev_sample(event):
        nonlocal current_idx
        current_idx = (current_idx - 1) % min(num_samples, len(dataset))
        update_plot()
    
    def print_info(event):
        img, lbl = dataset[current_idx]
        print(f"\n=== Sample {current_idx} Info ===")
        print(f"Image shape: {img.shape}")
        print(f"Label shape: {lbl.shape}")
        print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Label unique values: {torch.unique(lbl).tolist()}")
        if hasattr(dataset, 'pairs'):
            img_path, lbl_path = dataset.pairs[current_idx]
            print(f"Image file: {os.path.basename(img_path)}")
            print(f"Label file: {os.path.basename(lbl_path)}")
    
    # Create buttons
    ax_prev = plt.axes([0.2, 0.05, 0.1, 0.04])
    ax_next = plt.axes([0.7, 0.05, 0.1, 0.04])
    ax_info = plt.axes([0.45, 0.05, 0.1, 0.04])
    
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    btn_info = Button(ax_info, 'Print Info')
    
    btn_prev.on_clicked(prev_sample)
    btn_next.on_clicked(next_sample)
    btn_info.on_clicked(print_info)
    
    # Show first sample
    update_plot()
    plt.show()

# Alternative: Simple cycling with keyboard
def visualize_pairs_simple(dataset, num_samples=10):
    import matplotlib.pyplot as plt
    
    for i in range(min(num_samples, len(dataset))):
        img, lbl = dataset[i]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(img.squeeze(), cmap='gray')
        ax1.set_title(f'Sample {i}: Image')
        ax1.axis('off')
        
        ax2.imshow(lbl, cmap='tab10')
        ax2.set_title(f'Sample {i}: Label (unique: {torch.unique(lbl).tolist()})')
        ax2.axis('off')
        
        plt.suptitle(f'Press any key to continue to next sample...')
        plt.show()
        
        # Wait for user input
        input(f"Press Enter to continue to sample {i+1}...")


# print("Visualizing training samples:")
# train_ds = MRIDataset(train_pairs)
# visualize_pairs_interactive(train_ds, 100)  # Interactive version
# # visualize_pairs_simple(train_ds, 10)      # Simple version

def check_class_distribution(dataset, num_samples=10):
    print("Checking class distribution:")
    total_pixels = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in range(min(num_samples, len(dataset))):
        img, lbl = dataset[i]
        for c in range(4):
            class_counts[c] += (lbl == c).sum().item()
        total_pixels += lbl.numel()
    
    print(f"Total pixels: {total_pixels}")
    for c, count in class_counts.items():
        percentage = (count / total_pixels) * 100
        print(f"Class {c}: {count} pixels ({percentage:.1f}%)")

def visualize_predictions(model, test_loader, device, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(test_loader):
            if i >= num_samples:
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = torch.softmax(outputs, dim=1)
            
            # Show first image in batch
            img = imgs[0].cpu().squeeze()
            lbl = lbls[0].cpu()
            pred = preds[0].cpu()
            
            # Debug: print shapes
            print(f"Pred shape: {pred.shape}, Label shape: {lbl.shape}")
            print(f"Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"Pred channels: {pred.shape[0]}")
            print(f"Trying to access indices: {list(range(1, min(4, pred.shape[0])))}")
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 5 subplots for 5 images
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(lbl, cmap='tab10')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Show prediction for each class
            for c in range(1, 4):  # Classes 1, 2, 3 (LV, MYO, RV)
                axes[c+1].imshow(pred[c], cmap='hot')
                axes[c+1].set_title(f'Prediction Class {c}')
                axes[c+1].axis('off')
            
            plt.tight_layout()
            plt.show()

# Add this after training
# Load the best model for visualization


if __name__ == "__main__":
    """     
    Available model types:
    - 'unet': U-Net (baseline)
    - 'unetpp': U-Net++ (Nested U-Net)
    - 'manet': MAnet (Multi-scale Attention Network)
    - 'linknet': Linknet (Residual connections, no skip connections)
    - 'fpn': Feature Pyramid Network (multi-scale feature fusion)
    - 'pspnet': PSPNet (pyramid scene parsing with global context)
    - 'pan': PAN (pyramid attention with spatial attention)
    - 'deeplabv3': DeepLabV3 (atrous convolutions with ASPP)
    - 'deeplabv3plus': DeepLabV3+ (enhanced with decoder for fine details)
    - 'upernet': UPerNet (unified perceptual parsing with FPN)
    - 'segformer': Segformer (transformer-based hierarchical encoder)
    - 'dpt': DPT (dense prediction transformer with ViT backbone)"""
    # Train DeepLabV3+ with optimized hyperparameters
    try:
        # Best hyperparameters from tuning (you can update these based on your results)
        best_params = {
            'learning_rate': 0.001,
            'batch_size': 8,
            'epochs': 50,
            'weight_decay': 0.0001,
            'optimizer': 'adam',
            'class_weights': [1.0, 10.0, 10.0, 10.0]
        }
        
        print("Training DeepLabV3+ with optimized hyperparameters:")
        print(f"  Learning Rate: {best_params['learning_rate']}")
        print(f"  Batch Size: {best_params['batch_size']}")
        print(f"  Epochs: {best_params['epochs']}")
        print(f"  Weight Decay: {best_params['weight_decay']}")
        print(f"  Optimizer: {best_params['optimizer']}")
        print(f"  Class Weights: {best_params['class_weights']}")
        
        train_model(
            model_type='deeplabv3plus',
            learning_rate=best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            weight_decay=best_params['weight_decay'],
            class_weights=best_params['class_weights'],
            optimizer_type=best_params['optimizer']
        )
        
        print(f"\n🎵 Training completed with optimized parameters!")
        play_completion_sound()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        play_completion_sound()
    
    # For visualization (uncomment after training):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = DeepLabV3Plus(n_classes=4).to(device)
    # model.load_state_dict(torch.load('best_deeplabv3plus.pth'))
    # train_loader, val_loader, test_loader = get_loaders()
    # visualize_predictions(model, test_loader, device, num_samples=5)
