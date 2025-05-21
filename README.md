# Astronomical Image Autoencoder

This repository contains a deep learning model for compressing and reconstructing astronomical images using a convolutional autoencoder with U-Net architecture.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Latent Space Exploration](#latent-space-exploration)


## Project Overview
This project implements a neural network autoencoder specifically designed for astronomical image data. The autoencoder compresses FITS (Flexible Image Transport System) astronomical images into a compact 8-dimensional latent space representation and then reconstructs them with high fidelity.

### Project Overview Diagram
![project_overview_flowchart](https://github.com/user-attachments/assets/c515de53-a1e8-47ae-82f3-f8fb36d0cb14)

The project has applications in:
- **Data compression** for large astronomical datasets
- **Anomaly detection** for identifying unusual celestial objects
- **Feature extraction** for downstream machine learning tasks
- **Denoising** astronomical images
- **Exploring latent space representations** of astronomical data

## Features
- Custom data loader for FITS astronomical images
- U-Net architecture optimized for astronomical image reconstruction
- Comprehensive evaluation metrics (MSE, SSIM, MS-SSIM)
- Latent space visualization and analysis tools
- Customizable inference pipeline for testing


## Installation
To run this project, install the required dependencies:

```bash
 pip install numpy matplotlib torch torchvision astropy scikit-image tqdm pytorch-msssim
```

## Dataset
The model is trained on FITS astronomical image data. FITS is a standard format in astronomy for storing images and multi-dimensional data.

### Sample Image
![Modern Square Typographic Fashion Brand Logo](https://github.com/user-attachments/assets/5c66aa0f-d207-42a7-89d8-dd9c31263785)



The dataset preprocessing includes:
- Loading FITS files with variable dimensionality
- Resizing to a consistent 187×187 resolution
- Normalizing pixel values to [0,1] range
- Applying data augmentation (random flips and rotations) during training

## Model Architecture
The autoencoder is based on a U-Net architecture with customizations for astronomical data:

### U-Net Architecture
![unet_architecture_diagram](https://github.com/user-attachments/assets/723ed3ef-e555-4b42-bf0a-b0bf677f926f)

#### Encoder
- Four convolutional blocks with progressively increasing filters (16→32→64→128)
- Each block contains two convolutional layers with batch normalization and ReLU activations
- Max pooling for downsampling
- Final fully connected layer to create the latent space representation (8 dimensions)

#### Decoder
- Fully connected layer to reshape from latent space
- Series of upsampling blocks with skip connections from the encoder
- Each block contains transposed convolutions and regular convolutions
- Final sigmoid activation for output normalization

#### Skip Connections
- Preserve spatial information from encoder to decoder
- Help maintain fine details in the reconstructed images

## Training
The model was trained using:
- **Loss function:** Binary Cross-Entropy
- **Optimizer:** Adam with learning rate 0.001
- **Batch size:** 16
- **Epochs:** 150 (checkpoints every 10 epochs)
- **Train/Validation split:** 80/20

### Training Progress
![{744BC49D-FE7F-4EB9-B4CE-4CB14B315FBD}](https://github.com/user-attachments/assets/4138f8c2-7eac-4874-a544-94c08f6efd8c)

## Results
The trained model achieves excellent reconstruction quality:

### Reconstruction Examples
![{CC6568F6-C64A-489B-A65D-C37A59989778}](https://github.com/user-attachments/assets/39ef4a15-d1e6-40ff-8a22-e3ad40b297bc)


### Quantitative Metrics
- **Mean Squared Error (MSE):** 0.000054 (lower is better)
- **Structural Similarity Index (SSIM):** 0.9868 (higher is better)
- **Multi-Scale SSIM (MS-SSIM):** 0.9982 (higher is better)

### Metrics Visualization
![{DD1792FD-FAB9-4AB3-905F-9078ABFB8C1F}](https://github.com/user-attachments/assets/c97dfe4e-c7c6-4fdd-8077-ea71efc4421c)

## Latent Space Exploration
The 8-dimensional latent space captures key features of astronomical images:

### Latent Space Visualization
![{D48814B5-753D-4F20-953F-2E5E528EC8BE}](https://github.com/user-attachments/assets/3736283f-c099-40dc-a276-a868b4ff71fe)






