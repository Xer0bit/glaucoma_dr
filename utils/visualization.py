import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred_classes, classes):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

def get_gradcam_heatmap(model, img_array, last_conv_layer_index=-4):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.Model([model.inputs], [model.layers[last_conv_layer_index].output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return heatmap

def visualize_gradcam(model, sample_img_path, img_size, preprocess_image_func):
    """Generates and saves a Grad-CAM visualization for a sample image."""
    sample_img_orig = cv2.imread(sample_img_path)
    if sample_img_orig is None:
        print(f"Error: Could not read image at {sample_img_path}")
        return
    # Preprocess the image using the provided function for model input
    sample_img_processed = preprocess_image_func(sample_img_orig.copy())
    sample_img_array = np.expand_dims(sample_img_processed, axis=0)

    heatmap = get_gradcam_heatmap(model, sample_img_array)

    # Load the original image again for visualization (without normalization)
    # Resize the original image for overlay
    sample_img_display = cv2.resize(sample_img_orig, img_size)
    # Ensure the display image is in RGB if read by OpenCV (BGR)
    sample_img_display = cv2.cvtColor(sample_img_display, cv2.COLOR_BGR2RGB)


    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure heatmap is also RGB for combining
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap * 0.4 + sample_img_display
    superimposed_img = np.clip(superimposed_img, 0, 255) # Clip values to be in range [0, 255]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img_display)
    plt.title('Original Image')
    plt.axis('off') # Hide axes
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.title('Grad-CAM Heatmap')
    plt.axis('off') # Hide axes
    plt.tight_layout() # Adjust layout
    plt.savefig('gradcam_example.png')
    plt.close()