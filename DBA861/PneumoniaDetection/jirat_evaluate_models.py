#*****************************************************************************
# Filename: jirat_evaluate_models.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For evaluating trained models of pneumonia detection
#*****************************************************************************

#-- Import libraries ---------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import os

import config
#-----------------------------------------------------------------------------

#-- Function to evaluate accuracy of a model ---------------------------------
def evaluate_model(model_path):
    try:
        # Load the trained model
        print(f"\n-- Loading Model --")
        model = tf.keras.models.load_model(model_path)
        
        print(f"\nThe current location is: {os.path.basename(os.getcwd())}")
        print(f"\nThe path to the model is: {model_path}")  

        # Prepare the test data generator
        print(f"\n-- Prepare test data --")
        test_datagen = ImageDataGenerator(rescale=1.0/255)  # Rescale pixel values to [0, 1]

        test_generator = test_datagen.flow_from_directory(
            config.DATA_ROOT+config.TEST_FOLDER,  # Replace with the directory of the test dataset
            target_size=(224, 224),  # Resize images to match model input
            batch_size=32,  # Batch size for evaluation
            class_mode='categorical',  # Multi-class labels (categorical format)
            shuffle=False  # Ensure predictions match the order of test data
        )

        # Evaluate the model
        print(f"\n-- Evaluating the model on the test dataset --")
        test_loss, test_accuracy = model.evaluate(test_generator)

        # Generate predictions
        print(f"\nGenerating predictions on the test dataset...")
        y_pred = model.predict(test_generator)  # Predict probabilities for test data
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
        y_true = test_generator.classes  # True class labels from the test data

        # Generate and print classification report and confusion matrix
        print(f"\n\nModel: [{model_path}]")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
        print("\nClassification Report:")
        target_names = list(test_generator.class_indices.keys())  # Class names
        print(classification_report(y_true, y_pred_classes, target_names=target_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred_classes)
        print(cm)
    except Exception as e:
        print(f"\n\n[ERROR]:\b{e}")
#-- End of predict_image() ---------------------------------------------------

#----------------------------------------------------------------------------- 
# Main function
#----------------------------------------------------------------------------- 
if __name__ == "__main__":
#-- Evaluate CNN --
    # cnn = config.MODELS_ROOT+config.CNN
    # evaluate_model(cnn)
    
    #-- Evaluate VGG19 --
    print("------------------------------------------------------------")
    print(f"Evaluating {config.VGG19}")
    vgg19 = config.MODELS_ROOT+config.VGG19
    evaluate_model(vgg19)
    print("============================================================")

    #-- Evaluate ResNet50 --
    print("------------------------------------------------------------")
    print(f"Evaluating {config.RESNET50}")
    resnet50 = config.MODELS_ROOT+config.RESNET50
    evaluate_model(resnet50)
    print("============================================================")

    #-- Evaluate ResNet50V2 --
    print("------------------------------------------------------------")
    print(f"Evaluating {config.RESNET50V2}")
    resnet50V2 = config.MODELS_ROOT+config.RESNET50V2
    evaluate_model(resnet50V2)
    print("============================================================")
        
#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************