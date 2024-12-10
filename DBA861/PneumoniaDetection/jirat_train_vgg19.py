#*****************************************************************************
# Filename: jirat_train_resnet50.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For pneumonia detecter using ResNet50
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Main function
#----------------------------------------------------------------------------- 
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    # Set parameters
    input_shape = (224, 224, 3)  # Resize chest X-ray images to 224x224
    num_classes = 2  # Normal and Pneumonia
    batch_size = 32

    # Load the pre-trained ResNet50 model without the top layer
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the convolutional base
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Prepare data generators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        'D:\projects\GGU\data\chest_xray/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        'D:\projects\GGU\data\chest_xray/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_datagen.flow_from_directory(
        'D:\projects\GGU\data\chest_xray/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=10
    )

    # Fine-tune the model
    for layer in base_model.layers:
        layer.trainable = True  # Unfreeze all layers for fine-tuning

    # Recompile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Continue training
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=10
    )

    # Save the trained model
    model.save(f"./models/chest_xray_model_resnet50.h5")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # End the timer
    end_time = time.time()

    # Calculate the time elapsed
    elapsed_time = end_time - start_time

    # Log the time elapsed
    print(f"----- Time elapsed: {elapsed_time:.2f} seconds -----")

#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************