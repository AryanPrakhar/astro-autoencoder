
# Astrophysical Autoencoder: Deep Learning for Astronomical Image Compression

A PyTorch-based U-Net autoencoder designed to compress and analyze astrophysical images from FITS files. This project leverages deep learning to extract meaningful latent representations of astronomical observations while preserving critical spatial information. The latent representation could be used for downstream tasks like clustering, anomaly detection etc.

## Results

<img width="1039" height="625" alt="image" src="https://github.com/user-attachments/assets/dba1dea0-5282-4e00-9d17-fd74d36b27ac" />

Sample Output #1

<img width="1044" height="448" alt="image" src="https://github.com/user-attachments/assets/ed5cc6bb-b1b7-413d-aa42-3a44effc9eb3" />

Sample Output #2

<img width="1044" height="446" alt="image" src="https://github.com/user-attachments/assets/e0812bda-8bd2-49bd-8249-d14a700cafbe" />

Sample Output #3

<img width="1035" height="449" alt="image" src="https://github.com/user-attachments/assets/121e29cd-9990-4c60-88f0-d247b1afbd74" />


## Architecture

<img width="636" height="548" alt="image" src="https://github.com/user-attachments/assets/3b7228d4-1c24-40f4-98d4-5051f27f8c5c" />


## 🌟 Project Overview

This repository contains an end-to-end pipeline for:
- **Data Processing**: Loading and normalizing astrophysical FITS images
- **Model Training**: U-Net encoder-decoder architecture with an 8-dimensional latent space
- **Evaluation**: Comprehensive metrics including MSE, SSIM, and MS-SSIM
- **Analysis**: Latent space visualization using PCA and t-SNE
- **Inference**: Reconstruction and export of astronomical data

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ASTRO-AUTOENCODER PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    Install Libraries
                  (numpy, torch, etc.)
                                │
                                ▼
                    Data Preparation
              • Load FITS files
              • Resize & Normalize (187×187)
                                │
                                ▼
                    Model Definition
              • U-Net Encoder
              • Latent Space (8D)
              • U-Net Decoder
                                │
                                ▼
                    Training Phase
              • 80/20 Train/Validation Split
              • BCE Loss + Adam Optimizer
              • 150 Epochs
                                │
                                ▼
                    Evaluation Metrics
              • MSE, SSIM, MS-SSIM
              • Visualize Reconstructions
                                │
                                ▼
                    Latent Space Analysis
              • Extract Latent Vectors
              • PCA/t-SNE Visualization
                                │
                                ▼
                    Inference & Export
              • Test Data Reconstruction
              • Export Metrics & Results
```

## 🚀 Key Features

### Model Architecture
- **Encoder**: Progressive downsampling with skip connections
  - Input: 187×187 grayscale images
  - Channels: 1 → 16 → 32 → 64 → 128
  - Output: 8-dimensional latent vector

- **Decoder**: Progressive upsampling with skip connections
  - Reconstructs from 8D latent space
  - Restores spatial dimensions to 187×187
  - Output: Sigmoid-normalized grayscale image

### Technical Specifications
- **Framework**: PyTorch
- **Optimizer**: Adam
- **Loss Function**: Binary Cross Entropy (BCE)
- **Batch Normalization**: Applied throughout
- **Regularization**: Dropout (0.1)
- **Input Normalization**: Z-scale normalization with percentile clipping

### Evaluation Metrics
- **MSE** (Mean Squared Error): Pixel-level reconstruction accuracy
- **SSIM** (Structural Similarity Index): Perceptual quality preservation
- **MS-SSIM** (Multi-Scale SSIM): Multi-scale structural similarity


## 📦 Installation

1. **Clone the repository** (or set up the local directory):
```bash
git clone https://github.com/yourusername/astro-autoencoder.git
cd astro-autoencoder
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 🎯 Usage

### Running the Notebook

1. **Install Jupyter** (if not already installed):
```bash
pip install jupyter
```

2. **Launch Jupyter**:
```bash
jupyter notebook astro-autoencoder.ipynb
```

3. **Execute cells sequentially**:
   - Cell 1: Configure data directories
   - Cell 2: Install required libraries
   - Cell 3: Visualize sample FITS images
   - Cell 4-5: Define model architecture
   - Cell 6-7: Train the autoencoder
   - Cell 8-9: Evaluate performance
   - Cell 10-11: Analyze latent space
   - Cell 12-13: Run inference on test data


### Key Parameters

In the notebook, you can customize:

```python
# Training parameters
batch_size = 32
num_epochs = 150
learning_rate = 1e-3
latent_dim = 8
dropout_rate = 0.1

# Validation split
train_split = 0.8
val_split = 0.2

# Model architecture
# Encoder channels: 1 → 16 → 32 → 64 → 128
# Decoder channels: 128 → 64 → 32 → 16 → 1
```
