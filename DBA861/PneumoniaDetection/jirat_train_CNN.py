#*****************************************************************************
# Filename: jirat_train_cnn.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For pneumonia detecter using non-transfer learning CNN
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import time
from datetime import timedelta

import config
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Main function
#----------------------------------------------------------------------------- 
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    # Set parameters
    input_shape = (224, 224, 3)  # Resize chest X-ray images to 224x224
    batch_size = 32

    # Define a simple CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Assuming 2 classes: Normal and Pneumonia
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        config.DATA_ROOT+config.TRAINING_FOLDER,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        config.DATA_ROOT+config.VALIDATION_FOLDER,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_datagen.flow_from_directory(
        config.DATA_ROOT+config.TEST_FOLDER,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Set up an early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[early_stopping]
    )

    # Save the trained model
    model.save(config.MODELS_ROOT+config.CNN)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # End the timer
    end_time = time.time()

    # Calculate the time elapsed
    elapsed_time = end_time - start_time

    # Log the time elapsed
    print(f"----- Time elapsed: {str(timedelta(seconds=elapsed_time))} -----")

#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************