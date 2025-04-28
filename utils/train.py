import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)

def train_model(model, train_generator, val_generator, epochs):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint('hrf_deep_cnn_best.h5', monitor='val_accuracy', save_best_only=True)
    
    logger.info("Starting model training for %d epochs", epochs)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint]
    )
    
    return history

def evaluate_model(model, val_generator, classes):
    logger.info("Evaluating model performance")
    val_generator.reset()
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes
    
    report = classification_report(y_true, y_pred_classes, target_names=classes)
    logger.info("Classification Report:\n%s", report)
    print("Classification Report:")
    print(report)
    
    return y_true, y_pred_classes
