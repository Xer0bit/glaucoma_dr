import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(img_size, num_classes=3):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01), input_shape=(*img_size, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 5
        Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Classification head
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
