# Deep Learning Image Classification Project

This project implements a deep learning model for image classification using TensorFlow and Keras.

## Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.9
- NVIDIA GPU (recommended) with CUDA support
- CUDA Toolkit and cuDNN (for GPU acceleration)

## Hardware Requirements
- Minimum 8GB RAM
- NVIDIA GPU with at least 4GB VRAM (recommended)
- 10GB free disk space

## Setting up the Environment

1. Clone the repository
```bash
git clone <repository-url>
cd Vaneeza
```

2. Create and activate Conda environment
```bash
# Create new environment with GPU support
conda create -n vaneeza python=3.9
conda activate vaneeza

# Install CUDA toolkit and cuDNN (if using GPU)
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
Vaneeza/
├── data/                    # Dataset directory
├── models/                  # Saved model files
├── utils/                   # Utility functions
│   ├── data_preprocessing.py
│   ├── logging_config.py
│   └── train.py
├── main.py                  # Main training script
├── requirements.txt         # Project dependencies
└── README.md
```

## Features
- Deep CNN architecture for image classification
- GPU acceleration support
- Data augmentation pipeline
- Automated image preprocessing
- Training progress visualization
- Model performance evaluation

## Usage

1. Prepare your dataset:
   - Place your images in the `data/` directory
   - Organize images into subdirectories by class

2. Configure training parameters (optional):
   - Modify batch size, epochs, and learning rate in `main.py`
   - Adjust data augmentation settings in `utils/data_preprocessing.py`

3. Start training:
```bash
python main.py
```

## Model Architecture

The model implements a deep Convolutional Neural Network (CNN) with the following architecture:

### Network Structure
- Input Shape: 224 x 224 x 3 (RGB images)
- 5 Convolutional blocks with increasing filters
- Global Average Pooling
- Dense layers for classification

### Layer Details

1. **Block 1** - Initial Feature Extraction
   - 2x Conv2D (32 filters, 3x3 kernel)
   - Batch Normalization after each Conv2D
   - MaxPooling (2x2)
   - Dropout (25%)

2. **Block 2** - Feature Development
   - 2x Conv2D (64 filters, 3x3 kernel)
   - Batch Normalization after each Conv2D
   - MaxPooling (2x2)
   - Dropout (25%)

3. **Block 3** - Complex Feature Detection
   - 2x Conv2D (128 filters, 3x3 kernel)
   - Batch Normalization after each Conv2D
   - MaxPooling (2x2)
   - Dropout (30%)

4. **Block 4** - High-Level Feature Extraction
   - 2x Conv2D (256 filters, 3x3 kernel)
   - Batch Normalization after each Conv2D
   - MaxPooling (2x2)
   - Dropout (30%)

5. **Block 5** - Fine-Grained Feature Detection
   - 2x Conv2D (512 filters, 3x3 kernel)
   - Batch Normalization after each Conv2D
   - MaxPooling (2x2)
   - Dropout (40%)

6. **Classification Head**
   - Global Average Pooling
   - Dense layer (512 units)
   - Dropout (50%)
   - Final Dense layer (num_classes units)

### Key Features
- **L2 Regularization**: Applied to all Conv2D and Dense layers (0.01)
- **Batch Normalization**: After each convolution for stable training
- **Progressive Dropout**: Increasing from 25% to 50% deeper in the network
- **Global Average Pooling**: Reduces parameters and spatial information
- **Optimizer**: Adam with 1e-4 learning rate
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

### Design Choices
- Progressive increase in filters (32→64→128→256→512)
- Dense dropout layers to prevent overfitting
- Batch normalization for faster training
- L2 regularization for weight decay
- Small learning rate for fine-tuning

## Performance
- Training automatically uses GPU if available
- Progress bars show training status
- Training history plots are generated
- Confusion matrix visualization
- Classification report with precision, recall, and F1-score

## Development

### Managing the Environment
```bash
# Update dependencies
pip freeze > requirements.txt

# Remove environment if needed
conda deactivate
conda env remove -n vaneeza
```

### Logging
- Training progress is logged to console
- Detailed debug logs available in `logs/debug.log`
- Model checkpoints saved in `models/` directory

## Troubleshooting

### GPU Issues
1. Verify GPU is detected:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

2. Common fixes:
   - Update GPU drivers
   - Reinstall TensorFlow with GPU support
   - Check CUDA and cuDNN compatibility

## License
[Add your license information here]

## Contact
[Add your contact information here]
