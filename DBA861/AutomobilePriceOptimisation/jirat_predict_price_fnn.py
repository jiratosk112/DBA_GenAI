#*****************************************************************************
# Filename: jirat_predict_price_fnn.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For evaluating the trained FNN model
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import joblib

import config
#----------------------------------------------------------------------------- 


# Manually define numeric and categorical features
numeric_features = ['symboling', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
       'engine-size', 'compression-ratio', 'city-mpg', 'highway-mpg']

categorical_features = ['normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
       'body-style', 'drive-wheels', 'engine-location', 'engine-type',
       'num-of-cylinders', 'fuel-system', 'bore', 'stroke', 'horsepower',
       'peak-rpm']

# Evaluate the model accuracy
def evaluate_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.2f}, Test MAE: {test_mae:.2f}")
    return test_loss, test_mae

# Predict the optimal price
def predict_optimal_price(model, preprocessor, new_car_features):
    # Convert new car features to DataFrame
    new_car_df = pd.DataFrame([new_car_features])
    
    # Ensure data consistency with training
    new_car_df[numeric_features] = new_car_df[numeric_features].astype(float)
    new_car_df[categorical_features] = new_car_df[categorical_features].astype(str)
    
    # Preprocess new car features
    new_car_transformed = preprocessor.transform(new_car_df)
    
    # Make a prediction
    predicted_price = model.predict(new_car_transformed)
    print(f"Predicted Optimal Price for the New Car: ${predicted_price[0][0]:.2f}")
    return predicted_price[0][0]

#----------------------------------------------------------------------------- 
# Main function
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    # Evaluate FNN models
    print("\n-------------------------------------------------------------")
    print(f"\nEvaluate {config.FNN}")
    print("\n-------------------------------------------------------------")
    
    # Load test data and preprocessor
    X_test = joblib.load(config.MODELS_ROOT + config.X_TEST_FNN)
    y_test = joblib.load(config.MODELS_ROOT + config.Y_TEST_FNN)
    preprocessor = joblib.load(config.MODELS_ROOT + config.PREPROCESSOR_FNN)
    
    # Load trained FNN Model
    fnn_model = load_model(config.MODELS_ROOT + config.FNN)
    
    # Evaluate model, predict a price using test data, and print the result
    evaluate_model(fnn_model, X_test, y_test)
    predict_optimal_price(fnn_model, preprocessor, config.new_car_features)
    print("\n=============================================================")
    
#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************