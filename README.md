# deepfake-image-detection-AI-vs-Real
# Deepfake Detection Model

This project aims to develop a deep learning model to detect deepfake images by leveraging a combination of pre-trained Xception model and custom layers for texture extraction. The model classifies images as either "REAL" or "FAKE" based on learned features.
dataset: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images
## Project Structure

- **Data**: The dataset consists of real and fake images used for training and testing the model.
- **Model**: The architecture uses a pre-trained Xception model for feature extraction and additional layers for texture feature extraction and classification.
- **Training**: The model is fine-tuned on the dataset for deepfake detection.

## Model Architecture

### 1. Input Layer:
- **Shape**: `(128, 128, 3)`
- The model expects images of size 128x128 with 3 color channels (RGB).

### 2. Xception Base Model (Pre-trained on ImageNet):
- **Purpose**: The Xception model is used as a base for feature extraction. The top classification layers are excluded as they are not needed for this task.
- **Frozen Layers**: The initial layers of Xception are frozen to retain the learned features from ImageNet. Only the last 10 layers are trainable for fine-tuning.

### 3. Texture Extraction Layers:
- **Conv2D (32 filters) → MaxPooling2D**: A set of convolutional layers (32 filters) for texture feature extraction, followed by max pooling.
- **Conv2D (64 filters) → MaxPooling2D**: A second set of convolutional layers (64 filters) for further texture extraction, followed by max pooling.
- These layers help capture fine-grained texture details that are useful for distinguishing deepfakes.

### 4. Global Pooling:
- **GlobalAveragePooling2D**: Reduces the dimensionality of the feature maps by averaging spatial dimensions, which helps prevent overfitting.

### 5. Dense Layers:
- **Dense Layer (256 units)**: Fully connected layer with ReLU activation to learn complex patterns from the extracted features.
- **Dropout Layer (50%)**: Dropout regularization to prevent overfitting by randomly setting half of the neurons to zero during training.

### 6. Output Layer:
- **Dense Layer (1 unit with sigmoid activation)**: The final layer for binary classification (either "REAL" or "FAKE"), using sigmoid activation to output a probability between 0 and 1.

## Requirements

- Python 3.x
- TensorFlow (for model training and inference)
- Keras
- NumPy
- Matplotlib (for visualizations)

To install the required dependencies, you can use:

```bash
pip install -r requirements.txt
