#*****************************************************************************
# Filename: jirat_train_fnn.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For automobile pricing optimiser using FNN
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf

import joblib  # For saving the preprocessor

import time
from datetime import timedelta

import config
#----------------------------------------------------------------------------- 

#----------------------------------------------------------------------------- 
# Main function
#----------------------------------------------------------------------------- 
if __name__ == "__main__":
    print("\n-------------------------------------------------------------")
    print(f"\nTrain FNN Model: {config.FNN}")
    print("\n-------------------------------------------------------------")
    
    # Load data
    data = pd.read_csv(config.DATA_FILE)  # Load the dataset from the configured file path

    # Start the timer
    start_time = time.time()

    #-- Preprocessing Steps --
    # Replace '?' with NaN for easier handling of missing values
    data.replace('?', np.nan, inplace=True)  

    # Separate features and target variable
    X = data.drop('price', axis=1)  # Features (predictors)
    y = data['price']  # Target variable (price)

    # Handle target variable: Convert to numeric and handle missing values
    #   Convert target to numeric, replace invalid entries with NaN
    y = pd.to_numeric(y, errors='coerce')  
    #   Fill missing target values with the median to avoid bias
    y = y.fillna(y.median())  

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns  
    numerical_cols = X.select_dtypes(exclude=['object']).columns  

    # Preprocessing for numerical data: 
    #   Impute missing values with the mean and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler())  
    ])

    # Preprocessing for categorical data: 
    #   Impute missing values with the mode and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols), 
            ('cat', categorical_transformer, categorical_cols) 
        ])

    #-- Apply preprocessing and split data --    
    # Fit and transform the features
    X_preprocessed = preprocessor.fit_transform(X)  
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42) 

    #-- Convert to numpy arrays for TensorFlow compatibility --
    X_train = np.array(X_train.todense()) if hasattr(X_train, 
                                                     'todense') else X_train 
    X_test = np.array(X_test.todense()) if hasattr(X_test, 
                                                   'todense') else X_test
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #-- Build the FNN model --
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', 
                              input_shape=(X_train.shape[1],)),  
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(32, activation='relu'),  
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    #-- Compile the model --
    # Hyperparameters:
    # - Optimizer: Adam for adaptive learning rate optimization
    # - Loss: Mean Squared Error (suitable for regression problems)
    # - Metric: Mean Absolute Error for interpretability of prediction accuracy
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])

    # Train with validation split
    history = model.fit(X_train, y_train, 
                        epochs=100, 
                        validation_split=0.2, 
                        batch_size=32, 
                        verbose=1)  

    # Evaluate model performance on test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1) 
    print(f"\nTest Loss: {test_loss}, Test MAE: {test_mae}") 

    #-- Save the model for reuse --
    # Save test features for reuse
    joblib.dump(X_test, config.MODELS_ROOT + config.X_TEST_FNN)
    # Save test target for reuse  
    joblib.dump(y_test, config.MODELS_ROOT + config.Y_TEST_FNN)  

    # Save preprocessing pipeline for reuse
    joblib.dump(preprocessor, config.MODELS_ROOT + config.PREPROCESSOR_FNN)  
    # File path for saving the trained model
    trained_model_name = config.MODELS_ROOT + config.FNN  
    
    # Save the FNN model
    model.save(trained_model_name)  
    
    print(f"\nX_Test saved as {config.X_TEST_FNN}")
    print(f"y_Test saved as {config.Y_TEST_FNN}")
    print(f"Preprocessor saved as {config.PREPROCESSOR_FNN}")
    print(f"Model saved as {trained_model_name}")  
    
    # Calculate the time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n----- Time elapsed: {str(timedelta(seconds=elapsed_time))} -----")
    
    print("\n=============================================================")
#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************
