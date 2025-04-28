import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from utils.logging_config import setup_logging
logger = setup_logging()

logger.info("Starting application...")
logger.info("Importing dependencies...")

try:
    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    logger.error(f"Error importing TensorFlow: {e}")
    raise

import numpy as np
from utils import (
    create_data_generators, preprocess_image,
    create_model, train_model, evaluate_model,
    plot_training_history, plot_confusion_matrix, visualize_gradcam
)

# Set random seed for reproducibility
logger.info("Setting random seeds...")
tf.random.set_seed(42)
np.random.seed(42)

# Define parameters
data_dir = 'dataset' # Changed to 'hrf' to match comb.py, adjust if needed
img_size = (224, 224)
batch_size = 8
epochs = 50
classes = ['healthy', 'dr', 'glaucoma']
num_classes = len(classes)

def main():
    logger.info("Starting main execution...")
    
    logger.info("Creating data generators...")
    train_generator, val_generator = create_data_generators(data_dir, img_size, batch_size, classes)

    # Check if generators found images
    if not train_generator.samples:
        logger.error(f"No training images found in {data_dir}")
        return
    if not val_generator.samples:
        logger.error(f"No validation images found in {data_dir}")
        return

    logger.info("Creating and compiling model...")
    model = create_model(img_size, num_classes=num_classes)
    model.summary()

    logger.info("Starting model training...")
    history = train_model(model, train_generator, val_generator, epochs)

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    y_true, y_pred_classes = evaluate_model(model, val_generator, classes)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, classes)

    # Visualize Grad-CAM
    # Ensure the sample image path is correct and exists
    # Using the first image found in the 'glaucoma' directory of the validation set
    try:
        # Get the filenames and their corresponding classes from the validation generator
        filenames = val_generator.filenames
        class_indices = val_generator.class_indices
        idx_to_class = {v: k for k, v in class_indices.items()}

        # Find the first image belonging to the 'glaucoma' class
        sample_img_path = None
        for i, label_idx in enumerate(val_generator.classes):
            if idx_to_class[label_idx] == 'glaucoma':
                sample_img_path = os.path.join(data_dir, filenames[i])
                break

        if sample_img_path and os.path.exists(sample_img_path):
            print(f"Visualizing Grad-CAM for: {sample_img_path}")
            visualize_gradcam(model, sample_img_path, img_size, preprocess_image) # Pass preprocess_image
        else:
            print("Could not find a sample 'glaucoma' image in the validation set for Grad-CAM.")

    except Exception as e:
        print(f"Error during Grad-CAM visualization: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in main execution")
        raise
