# MRAI-Tumor Segmentation & 3D-Export Engine

**Developer:** Tair Fridman | B.Sc. Biomedical Engineering, Technion

## Overview<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5d9e7c40-4e3d-4645-8c83-6e9b8edcc042" />

## Overview
The **MRAI Engine** is a comprehensive medical imaging platform designed for high-precision neuroimaging analysis. It bridges the gap between raw radiological data and actionable clinical insights by integrating a custom-trained **3D Fully Convolutional Network (FCN)** for automated tumor segmentation. 

Unlike standard viewers, this system performs **True-Scale Inference** on 1:1 voxel data, preserving spatial integrity for accurate **3D volumetric calculations** and **STL mesh generation**. From advanced frequency-domain signal processing (FFT) to automated clinical reporting, the MRAI Engine provides a complete end-to-end pipeline for modern digital pathology and surgical planning.

## Key Features

* **Deep Learning Tumor Detection & Contouring:** Integrates a custom 3D U-Net architecture (Fully Convolutional Network) to automatically detect, localize, and draw precise clinical contours around brain tumors, complete with probability scoring.
* **Interactive Orthogonal Navigation:** Features a synchronized UI for simultaneous viewing of Axial, Coronal, and Sagittal planes. Includes real-time slicing, spatial crosshairs, deep zooming, and axis rotation.
* **3D STL Generation & True Volumetry:** Converts MRI pixel data into 3D printable STL meshes (capable of rendering both the full brain and isolated tumor masses). Calculates exact physical volume in real-world units (cm³) using DICOM/NIfTI spatial metadata.
* **Advanced Frequency-Domain DSP:** Implements a custom Gaussian High-Pass Filter via Fast Fourier Transform (FFT) to highlight fine edges and morphological details without ringing artifacts.
* **Real-Time Image Enhancement & Color Mapping:** Provides dynamic image sharpening tools and multiple colormap filters (e.g., Grayscale, Hot, Viridis, Inferno) to improve tissue contrast and visual diagnosis.
* **Multi-Format Clinical Data Pipeline:** Robust parsing engine capable of handling raw DICOM series and compressed NIfTI volumes, automatically calibrating physical voxel spacing and aspect ratios.
* **Automated Clinical PDF Reporting:** Instantly compiles current multi-planar snapshot views, AI diagnosis confidence, and volumetric calculations into a formatted, ready-to-print medical report.

## Tech Stack
* **Deep Learning:** PyTorch
* **Image Processing:** Scikit-Image, SciPy, NumPy, Nilearn
* **Medical Formats:** NiBabel, PyDICOM
* **GUI & Visualization:** Matplotlib, Tkinter, stl-mesh

## How to Run

This project requires Python 3.8+ and a standard 64-bit OS. 

1. **Download the project**
   Clone this repository or download it as a ZIP file and extract it.
   
2. **Install dependencies**
   Open a terminal in the project folder and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Viewer**
   Ensure the pre-trained model weights file (`brats_3d_unet_full.pth`) is in the same folder, then run the main application:
   ```bash
   python MriFINAL.py
   ```
