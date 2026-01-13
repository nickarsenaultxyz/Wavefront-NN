# Wavefront Set Neural Network - U-Net Training

A deep learning approach to predicting wavefront sets (surface normals/angles) of geometric shapes from binary boundary masks, with special handling for corner singularities.

## Overview

This project implements a U-Net-based neural network that predicts the **wavefront set** of geometric shapes. The wavefront set represents the directional information (surface normals) at boundary points. A key innovation of this implementation is its ability to handle **singularities at corners**, where a single boundary pixel may possess multiple valid normal directions.

Mathematically, the wavefront set, denoted $WF(u)$ is a subset of the cotangent bundle $T^* \mathbb{R}^n$. It contains points $(x_0,\xi_0)$ such that there exists a conic neighborhood $\Gamma$ of $\xi$ where $\mathcal{F}(u(x,\xi))$ is not rapidly decreasing. That is, there exists $\varphi \in C^\infty_c(\mathbb{R}^n)$, $\varphi(x_0) \neq 0$ such that
$$\forall N \in \mathbb{N} \exists C_n > 0 \text{ such that }  |\hat{\varphi u}| \leq C_N (1 + |\xi|)^{-N}$$
for all $\xi \in \Gamma$.
## Key Features

### Architecture
- **Custom U-Net**: Encoder-decoder architecture adapted for pixel-wise classification of angle bins
- **Multi-label Classification**: Uses BCEWithLogitsLoss to independently predict probabilities for each angle bin
- **36 Angle Bins**: Discretizes the angle space [0, π) into 36 bins for classification
- **~31M Parameters**: Deep network capable of learning complex geometric patterns

### Data Generation
- **Synthetic Shape Dataset**: Generates random geometric shapes with precise normal calculations
  - **Circles**: Radial normals
  - **Ellipses**: Non-radial normals accounting for major/minor axes
  - **Squares**: 4-sided polygons with corner detection
  - **Polygons**: Arbitrary n-sided shapes with variable vertices

- **Multi-Normal Support**: Specialized corner handling that stores multiple normal vectors at polygon vertices
  - Corner pixels: Both adjacent edge normals stored in channels 0 and 1
  - Edge pixels: Single normal in channel 0, sentinel value (-100) in channel 1
  - Enables multi-hot encoding for training

### Training Features
- **Multi-hot Target Encoding**: Target shape (K, H, W) allows multiple valid angles per pixel
- **Boundary Masking**: Loss computed only on boundary pixels
- **Data Augmentation**: Random positioning, sizing, and rotation of shapes
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning

## Results

### Training Performance
- **Training Dataset**: 2000 samples (1600 train, 400 validation)
  - 50% Circles
  - 50% Squares
- **Final Training Accuracy**: 90.3% (epoch 30)
- **Best Validation Accuracy**: 89.1% (epoch 26)
- **Best Validation Loss**: 0.5644

### Generalization
- **Polygon Test Accuracy**: 75.96% on 200 unseen pentagon shapes
- **Edge Detection**: High accuracy on straight edges and smooth curves
- **Primary Orientation**: Successfully identifies dominant edge orientations

### Corner Singularity Analysis
The model shows interesting behavior at corners:
- **Primary Normal**: Correctly identifies with high probability (~55%)
- **Secondary Normal**: Struggles to produce distinct bimodal distributions
- **Smoothing Bias**: Tends to predict neighboring bins rather than separated peaks
- **Future Work**: Enhanced corner detection may require architectural changes or specialized loss weighting

## Technical Details

### Input/Output
- **Input**: Binary mask (1, H, W) where H=W=128
- **Output**: Probability map (K, H, W) where K=36 angle bins
- **Angle Range**: [0, π) normalized and discretized

### Loss Function
```python
criterion = nn.BCEWithLogitsLoss(reduction='none')
# Masked to boundary pixels only
loss = (loss_map * masks).sum() / (masks.sum() + 1e-8)
```

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Size**: 16
- **Epochs**: 30
- **Device**: CUDA (GPU required for practical training)

## Dependencies

```
Python 3.x
torch >= 1.9.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scipy >= 1.5.0
```

## Usage

### 1. Setup Environment
```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### 2. Generate Training Data
```python
# Generate synthetic shapes with normals
num_samples = 2000
H, W = 128, 128

for i in range(num_samples):
    if i % 2 == 0:
        angle_map, mask = generate_circle_normals(H, W, seed=i)
    else:
        angle_map, mask = generate_square_normals(H, W, seed=i)
```

### 3. Train Model
```python
model = WavefrontUNet(K=36, bilinear=False)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop with validation
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch_multilabel(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_multilabel(model, val_loader, criterion, device)
```

### 4. Evaluate and Visualize
```python
# Load best model
model.load_state_dict(torch.load('best_wavefront_unet.pth'))
model.eval()

# Generate test sample
test_angles, test_mask = generate_polygon_normals(H, W, num_vertices=5, seed=42)

# Predict
with torch.no_grad():
    logits = model(mask_tensor)
    probs = torch.sigmoid(logits)
```

## File Structure

```
wavefront_unet_colab.ipynb
├── Section 1: Dependencies and Setup
├── Section 2: Data Generation Functions
│   ├── generate_circle_normals()
│   ├── generate_ellipse_normals()
│   ├── generate_square_normals()
│   └── generate_polygon_normals()
├── Section 3: Dataset Generation
├── Section 4: U-Net Model Definition
│   ├── DoubleConv, Down, Up layers
│   └── WavefrontUNet wrapper
├── Section 5: Dataset and DataLoader
├── Section 6: Training Functions
├── Section 7: Training Loop
├── Section 8: Training Visualization
└── Section 9: Testing and Evaluation
    ├── Polygon generalization test
    ├── Corner pixel analysis
    └── Batch evaluation
```

## Model Architecture

```
WavefrontUNet (K=36 output classes)
│
├── Encoder Path
│   ├── inc: DoubleConv(1 → 64)
│   ├── down1: MaxPool + DoubleConv(64 → 128)
│   ├── down2: MaxPool + DoubleConv(128 → 256)
│   ├── down3: MaxPool + DoubleConv(256 → 512)
│   └── down4: MaxPool + DoubleConv(512 → 1024)
│
├── Decoder Path
│   ├── up1: Upsample + DoubleConv(1024 → 512)
│   ├── up2: Upsample + DoubleConv(512 → 256)
│   ├── up3: Upsample + DoubleConv(256 → 128)
│   └── up4: Upsample + DoubleConv(128 → 64)
│
└── outc: Conv2d(64 → 36)
```

## Data Format

### Angle Maps (H, W, 2)
- **Channel 0**: Primary normal angle or first edge at corners
- **Channel 1**: Second edge normal at corners, sentinel (-100.0) otherwise
- **Valid Range**: [0, π) for angles, -100.0 for invalid

### Target Encoding (K, H, W)
- **Multi-hot**: Multiple bins can be 1.0 for corner pixels
- **Single-hot**: Only one bin is 1.0 for edge pixels
- **Zero**: All bins 0.0 for non-boundary pixels

## Evaluation Metrics

### Top-1 Accuracy
```python
pred_bins = torch.argmax(outputs, dim=1, keepdim=True)
is_correct = torch.gather(targets, 1, pred_bins)
accuracy = (is_correct[boundary_mask] > 0.5).mean()
```

Measures if the highest probability bin matches any ground truth angle (important for corners with multiple valid angles).

## Visualization

The notebook includes comprehensive visualization:
1. **Training Curves**: Loss and accuracy over epochs
2. **Prediction Comparison**: Input mask, ground truth, predicted angles
3. **Probability Distributions**: Per-pixel angle bin probabilities
4. **Corner Analysis**: Bimodal distribution analysis at singularities

## Known Limitations

1. **Corner Detection**: Model struggles to predict distinct bimodal distributions at corners, showing preference for smoothness
2. **Generalization**: 75.96% accuracy on polygons vs 89.1% on trained shapes (circles/squares)
3. **Resolution**: Fixed 128×128 resolution may limit fine detail capture
4. **Angle Discretization**: 36 bins (5° resolution) may be too coarse for some applications

## Future Improvements

1. **Architectural Changes**:
   - Attention mechanisms for corner detection
   - Multi-scale feature aggregation
   - Deeper networks or residual connections

2. **Training Enhancements**:
   - Corner pixel loss weighting
   - Contrastive learning for multi-modal outputs
   - Data augmentation with more shape variety

3. **Evaluation**:
   - Per-corner accuracy metrics
   - Bimodality detection scores
   - Edge vs corner performance breakdown

## Research Context

This project explores the application of deep learning to geometric singularity detection, specifically the wavefront set in singularity theory. The ability to detect multiple directions at corners has applications in:
- Computer vision (edge and corner detection)
- Shape analysis and reconstruction
- Geometric deep learning
- Medical image segmentation (detecting branching structures)
