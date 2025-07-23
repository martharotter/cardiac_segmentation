import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CardiacSegmentation2DDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_size=(256, 256)):
        self.samples = []
        self.target_size = target_size

        image_filenames = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.nii.gz')
        ])

        for img_fname in image_filenames:
            parts = img_fname.split('_')
            label_fname = '_'.join(parts[:3]) + '_gt_' + parts[3]

            img_path = os.path.join(image_dir, img_fname)
            lbl_path = os.path.join(label_dir, label_fname)

            image_3d = nib.load(img_path).get_fdata()
            label_3d = nib.load(lbl_path).get_fdata()

            # Store 2D slices and their matching labels
            for i in range(image_3d.shape[2]):  # loop over slices
                self.samples.append((image_3d[:, :, i], label_3d[:, :, i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_slice, lbl_slice = self.samples[idx]

        # Normalize and convert
        img_slice = (img_slice - np.mean(img_slice)) / np.std(img_slice)
        img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
        lbl_tensor = torch.tensor(lbl_slice, dtype=torch.long).unsqueeze(0)

        # Resize both image and label
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        lbl_tensor = F.interpolate(lbl_tensor.unsqueeze(0).float(), size=self.target_size, mode='nearest').squeeze(0).long()

        return img_tensor, lbl_tensor
