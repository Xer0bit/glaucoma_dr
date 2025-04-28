import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Found {len(gpus)} GPU(s). Using CUDA for training.")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
else:
    logger.warning("No GPU found. Training will use CPU.")

# Try importing OpenCV, fallback to PIL if not available
try:
    import cv2
    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'resize'):
        USE_OPENCV = True
        logger.info("Using OpenCV for image processing")
    else:
        USE_OPENCV = False
        logger.warning("OpenCV installation incomplete, falling back to PIL")
except ImportError:
    USE_OPENCV = False
    logger.warning("OpenCV not available, using PIL for image processing")

def ensure_uint8(img):
    """Ensure image is in uint8 format."""
    if img.dtype != np.uint8:
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img

def crop_borders(img, threshold=10):
    """Crop image borders using either OpenCV or PIL."""
    try:
        # Convert input to proper format
        if isinstance(img, np.ndarray):
            img = ensure_uint8(img)
        
        if USE_OPENCV:
            if len(img.shape) == 3:
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                except Exception:
                    gray = np.mean(img, axis=2).astype(np.uint8)
            else:
                gray = img.astype(np.uint8)
        else:
            # Convert to PIL Image if not already
            if not isinstance(img, Image.Image):
                img = Image.fromarray(ensure_uint8(img))
            gray = img.convert('L')
            gray = np.array(gray)

        mask = gray > threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
            
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        if USE_OPENCV:
            return img[x_min:x_max+1, y_min:y_max+1]
        else:
            img = np.array(img)
            return img[x_min:x_max+1, y_min:y_max+1]
            
    except Exception as e:
        logger.error(f"Error in crop_borders: {e}")
        return img

def preprocess_image(img):
    """Preprocess image using either OpenCV or PIL."""
    try:
        if img is None:
            logger.error("Input image is None")
            return None
            
        # Convert input to proper format
        if isinstance(img, np.ndarray):
            img = ensure_uint8(img)
            
        # Crop borders
        img = crop_borders(img)
        
        # Resize
        if USE_OPENCV:
            try:
                img = cv2.resize(ensure_uint8(img), (224, 224))
            except Exception as e:
                logger.warning(f"OpenCV resize failed: {e}, falling back to PIL")
                img = Image.fromarray(ensure_uint8(img))
                img = img.resize((224, 224))
                img = np.array(img)
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(ensure_uint8(img))
            img = img.resize((224, 224))
            img = np.array(img)
            
        # Normalize
        img = img.astype(np.float32) / 255.0
        return img
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        return None

def create_data_generators(data_dir, img_size, batch_size, classes, validation_split=0.2):
    """Creates training and validation data generators with augmentation."""


    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        preprocessing_function=preprocess_image,
        validation_split=validation_split
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=classes,
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes=classes,
        shuffle=False
    )

    return train_generator, val_generator
