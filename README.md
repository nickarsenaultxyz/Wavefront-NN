# Wavefront-NN
## Overview
This project implements a **U-Net** based neural network to predict the **wavefront set** (surface normals/angles) of geometric shapes given their binary boundary masks. A key feature of this implementation is its ability to handle **singularities** (corners), where a single boundary pixel may possess multiple valid normal directions.

## Key Features
*   **Custom U-Net Architecture**: Adapted for pixel-wise classification of angle bins.
*   **Synthetic Data Generation**:
    *   Generates random shapes: Circles, Ellipses, Squares, and arbitrary Polygons.
    *   **Multi-Normal Support**: Specifically designed to calculate and store multiple normal vectors at polygon vertices (corners).
*   **Multi-Label Classification**:
    *   Uses `BCEWithLogitsLoss` to independently predict probabilities for each angle bin.
    *   Allows the model to predict multiple angles for a single pixel (e.g., at corners).
*   **Corner Analysis**: Includes visualization and evaluation metrics specifically for detecting corner singularities.

## Dependencies
*   Python 3.x
*   PyTorch (CUDA recommended for training)
*   NumPy
*   Matplotlib
*   SciPy

## Usage
1.  **Setup**: Run the installation cell to check for GPU availability and import libraries.
2.  **Data Generation**: Execute the data generation cells to create synthetic training samples (Circles, Squares, etc.) with multi-channel angle maps.
3.  **Model Training**: Train the U-Net using the provided training loop. The model saves the best weights based on validation loss.
4.  **Evaluation**: Use the testing cells to visualize predictions on unseen polygons and analyze probability distributions at corner pixels.

## Model Details
*   **Input**: Binary mask of the shape boundary $(1, H, W)$.
*   **Output**: Multi-channel probability map $(K, H, W)$, where $K=36$ angle bins.
*   **Loss Function**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`).

## Results
*   The model achieves high accuracy (~80%+) on general boundary pixels.
*   It successfully identifies primary edge orientations.
*   Corner detection analysis reveals the model's behavior at singularities, providing a foundation for further research into high-frequency edge detection.
