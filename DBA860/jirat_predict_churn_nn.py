#*****************************************************************************
# Filename: jirat_predict_churn_nn.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For training a robust neural network model to predict churn
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

import matplotlib.pyplot as plt

import pickle

import config
#----------------------------------------------------------------------------- 

#----------------------------------------------------------------------------- 
# Define preprocess_data()
#----------------------------------------------------------------------------- 
def preprocess_data():
    try:
        # Load the dataset
        data = pd.read_csv(config.DATA_FILE)

        # Drop columns not relevant for modeling (e.g., "customerID")
        data = data.drop(["customerID"], axis=1)

        # Convert "TotalCharges" to numeric, handling errors (e.g., blanks)
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

        # Fill missing values in "TotalCharges" with the median value
        data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)
        
       # Convert categorical variables to numeric using OneHotEncoder and LabelEncoder
        categorical_features = data.select_dtypes(include=["object"]).columns
        categorical_features = categorical_features.drop("Churn")  # Exclude target variable

        # One-hot encode categorical features
        data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

        # Label encode the target variable "Churn" (Yes = 1, No = 0)
        data["Churn"] = LabelEncoder().fit_transform(data["Churn"])

        # Standardize numerical features
        numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Split the data into features (X) and target (y)
        X = data.drop("Churn", axis=1)
        y = data["Churn"]

        # Split the data into training and testing sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        return data, X, y, X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"\n[prep_data()] Data Preparation Failed!")
        raise
#-- End of preprocess_data() -------------------------------------------------

#----------------------------------------------------------------------------- 
# Define train_model()
#-----------------------------------------------------------------------------
def train_model(data, X_train, X_test, y_train, y_test):
    try:
        # Initialize the Neural Network model
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

        # Define callbacks: learning rate scheduler and early stopping
        def lr_scheduler(epoch, lr):
            if epoch > 10:
                return lr * 0.5
            return lr

        callbacks = [
            LearningRateScheduler(lr_scheduler, verbose=0),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        # Train the model on the training data
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Save the trained model to a file for future use
        model_filename = config.MODELS_ROOT + config.CHURN_NN_MODEL
        model.save(model_filename)
        
        return model, history
    except Exception as e:
        print(f"\n[train_model()] Model Training Failed!")
        raise
#-- End of train_model() -----------------------------------------------------

#----------------------------------------------------------------------------- 
# Define evaluate_model()
#-----------------------------------------------------------------------------
def evaluate_model(model, X, y, X_test, y_test):
    try:
        # Evaluate the model
        loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print evaluation metrics
        print()
        print(f"****************************************")
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"****************************************")
    except Exception as e:
        print(f"\n[evaluate_model()] Model Evaluation Failed!")
        raise
#-- End of evaluate_model() --------------------------------------------------

#----------------------------------------------------------------------------- 
# Define display_charts()
#----------------------------------------------------------------------------- 
def display_charts(history):
    try:
        # Visualization: Loss and Accuracy over epochs
        plt.figure(figsize=(12, 6))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise
#-- display_charts() ---------------------------------------------------------

#----------------------------------------------------------------------------- 
# Main function
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print(f"\n\n")
        print(f"------------------------------------------------------------")
        print(f"Churn Prediction - Model Training")
        print(f"------------------------------------------------------------")
        
        #-- Preprocessing --
        data, X, y, X_train, X_test, y_train, y_test = preprocess_data()
        
        #-- Model Training --
        model, history = train_model(data, X_train, X_test, y_train, y_test)
        
        #-- Model Evaluation --
        evaluate_model(model, X, y, X_test, y_test)
        
        #-- Model Visualization --
        display_charts(history)
        
        print(f"------------------------------------------------------------")
    except Exception as e:
        print(f"\n\n")
        print(f"************************************************************")
        print(f"Crash Exit Log")
        print(f"************************************************************")
        print(f"[ERROR]: {e}")
        print(f"============================================================")
        print(f"\n\n")
#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************
