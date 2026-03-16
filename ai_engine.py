import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from scipy.ndimage import binary_fill_holes

warnings.filterwarnings('ignore')

# ==========================================
# 1. 3D FCN Architecture Definition
# ==========================================
class DoubleConv3D(nn.Module):
    """
    Standard Double Convolution block for 3D U-Net architectures.
    Includes Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    """
    Fully Convolutional 3D U-Net.
    Capable of accepting arbitrary input dimensions (must be multiples of 8).
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.down1 = DoubleConv3D(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv3D(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        self.bottleneck = DoubleConv3D(64, 128)
        
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv3D(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv3D(64, 32)
        self.up3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv3D(32, 16)
        
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        
        d1 = self.up1(x4)
        d1 = torch.cat((x3, d1), dim=1)
        d1 = self.up_conv1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.up_conv2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat((x1, d3), dim=1)
        d3 = self.up_conv3(d3)
        
        out = self.out_conv(d3)
        return torch.sigmoid(out)

# ==========================================
# 2. Engine Initialization & Model Loading
# ==========================================
MODEL_PATH = 'brats_3d_unet_full.pth' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tumor_ai = None

try:
    if os.path.exists(MODEL_PATH):
        tumor_ai = UNet3D(in_channels=1, out_channels=1).to(device)
        tumor_ai.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        tumor_ai.eval() 
        print(f"--- 3D FCN AI Engine Ready on {device} ---")
    else:
        print(f"Error: Model weights '{MODEL_PATH}' not found in directory.")
except Exception as e:
    print(f"Error loading AI Engine: {e}")
    tumor_ai = None

# ==========================================
# 3. Full-Scale Inference Pipeline
# ==========================================
def scan_full_volume(volume_data, rot_k, progress_callback=None):
    """
    Performs a 1:1 scale scan of the MRI volume using the 3D FCN, 
    applying geometric filters to eliminate false positives.
    """
    orig_shape = volume_data.shape 
    if tumor_ai is None: 
        return False, 0.0, None, orig_shape[2] // 2, None, None

    if progress_callback: progress_callback(10, 100)
    
    # 1. Intensity Normalization
    v_min, v_max = volume_data.min(), volume_data.max()
    vol_norm = (volume_data - v_min) / (v_max - v_min) if v_max > v_min else volume_data
        
    # 2. Dynamic Padding for FCN compatibility (Multiples of 8)
    if progress_callback: progress_callback(30, 100)
    print("Padding volume for true-scale inference...")
    
    pad_x = (8 - orig_shape[0] % 8) % 8
    pad_y = (8 - orig_shape[1] % 8) % 8
    pad_z = (8 - orig_shape[2] % 8) % 8
    
    input_tensor = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    padded_tensor = F.pad(input_tensor, (0, pad_z, 0, pad_y, 0, pad_x), mode='constant', value=0).to(device)
    
    # 3. Neural Network Forward Pass
    if progress_callback: progress_callback(60, 100)
    print(f"Executing 3D Segmentation on dimensions: {padded_tensor.shape}...")
    with torch.no_grad():
        output_tensor = tumor_ai(padded_tensor)
        prob_vol_padded = output_tensor.squeeze().cpu().numpy()
        
    # 4. Crop padding back to original patient dimensions
    if progress_callback: progress_callback(80, 100)
    prob_vol_orig = prob_vol_padded[0:orig_shape[0], 0:orig_shape[1], 0:orig_shape[2]]
    
    # 5. Geometric and Morphological Filtering (Noise Reduction)
    raw_binary = (prob_vol_orig > 0.5)
    labels = measure.label(raw_binary)
    
    valid_blobs = []
    if labels.max() > 0:
        props = measure.regionprops(labels)
        for prop in props:
            # Clinical Geometry Filters:
            # Area > 50: Ignores micro-artifacts or tiny vessels
            # Solidity > 0.4: Rejects elongated structures (e.g., skull base, normal cortical folds)
            if prop.area > 50 and prop.solidity > 0.4:
                valid_blobs.append(prop)
                
    if not valid_blobs:
        mask_vol = np.zeros_like(raw_binary, dtype=np.uint8)
        tumor_pixels = 0
        confidence = 0.0
        has_tumor = False
        
        print("\n--- CLINICAL FILTER TRIGGERED ---")
        print("Reason: Candidate regions failed volumetric or morphological standards.")
        print("Result: No malignant mass detected.")
        print("---------------------------------\n")
        
        if progress_callback: progress_callback(100, 100)
        return False, 0.0, None, orig_shape[2] // 2, None, None
        
    else:
        # Isolate the primary mass based on physical volume
        best_blob = max(valid_blobs, key=lambda x: x.area)
        single_tumor_mask = (labels == best_blob.label)
        
        # Internal void filling
        mask_vol = binary_fill_holes(single_tumor_mask).astype(np.uint8)
        
        # Data Extraction
        max_prob = float(np.max(prob_vol_orig[mask_vol > 0]))
        confidence = max_prob * 100.0
        tumor_pixels = np.sum(mask_vol)
        has_tumor = True

    # 6. Anchor Point and Slice Extraction for UI Navigation
    z_sums = mask_vol.sum(axis=(0, 1))
    best_z = int(np.argmax(z_sums))
    z_indices = np.where(z_sums > 0)[0]
    tumor_range = (int(z_indices[0]), int(z_indices[-1]))
    
    raw_2d_mask = mask_vol[:, :, best_z]
    ai_mask_rotated = np.rot90(raw_2d_mask, k=rot_k)
    
    masked_slice = volume_data[:, :, best_z] * raw_2d_mask
    if np.any(masked_slice):
        max_idx = np.unravel_index(np.argmax(masked_slice), masked_slice.shape)
        anchor_x, anchor_y = int(max_idx[0]), int(max_idx[1])
    else:
        anchor_x, anchor_y = -1, -1

    if progress_callback: progress_callback(100, 100)
    print(f"AI ENGINE SUCCESS: Primary mass identified. Confidence: {confidence:.1f}%. Peak Anchor: Z={best_z}")
    return True, confidence, tumor_range, best_z, ai_mask_rotated, (anchor_x, anchor_y)