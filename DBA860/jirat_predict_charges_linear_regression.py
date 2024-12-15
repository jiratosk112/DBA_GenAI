#*****************************************************************************
# Filename: jirat_predict_charges_linear_regression.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For training a linear regression model to predict TotalCharges
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        
        # Convert categorical variables to numeric using OneHotEncoder
        categorical_features = data.select_dtypes(include=["object"]).columns

        # One-hot encode categorical features
        data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

        # Standardize numerical features
        numerical_features = ["tenure", "MonthlyCharges"]  # Exclude TotalCharges
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Split the data into features (X) and target (y)
        X = data.drop("TotalCharges", axis=1)
        y = data["TotalCharges"]

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
        # Initialize the Linear Regression model
        model = LinearRegression()

        # Train the model on the training data
        model.fit(X_train, y_train)
        
        # Save the trained model to a file for future use
        model_filename = config.MODELS_ROOT + config.LINEAR_REGRESSION_MODEL
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        
        return model
    except Exception as e:
        print(f"\n[train_model()] Model Training Failed!")
        raise
#-- End of train_model() -----------------------------------------------------

#----------------------------------------------------------------------------- 
# Define evaluate_model()
#----------------------------------------------------------------------------- 
def evaluate_model(model, X, y, X_test, y_test):
    try:
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Print coefficients to interpret feature influence
        coefficients = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        })
        print()
        print(f"*******************************************************")
        print(coefficients.sort_values(by="Coefficient", ascending=False))
        print(f"*******************************************************")

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print evaluation metrics
        print()
        print(f"****************************************")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-Squared (R2): {r2}")
        print(f"****************************************")
    except Exception as e:
        print(f"\n[evaluate_model()] Model Evaluation Failed!")
        raise
#-- End of evaluate_model() --------------------------------------------------

#----------------------------------------------------------------------------- 
# Define display_charts()
#----------------------------------------------------------------------------- 
def display_charts(model, X, y, X_test, y_test):
    try:
        # Get predicted values for the test data
        y_pred = model.predict(X_test)

        # Visualization: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
        
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.title("Actual vs Predicted TotalCharges")
        plt.xlabel("Actual TotalCharges")
        plt.ylabel("Predicted TotalCharges")
        plt.show()
        
        # Visualization: Coefficients
        coefficients = pd.Series(model.coef_, index=X.columns).sort_values()
        plt.figure(figsize=(12, 10))
        coefficients.plot(kind="barh", color="teal")
        plt.title("Feature Importance (Coefficients)")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Features")
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
        print(f"TotalCharges Prediction - Model Training")
        print(f"------------------------------------------------------------")
        
        #-- Preprocessing --
        data, X, y, X_train, X_test, y_train, y_test = preprocess_data()
        
        #-- Model Training --
        model = train_model(data, X_train, X_test, y_train, y_test)
        
        #-- Model Evaluation --
        evaluate_model(model, X, y, X_test, y_test)
        
        #-- Model Visualization --
        display_charts(model, X, y, X_test, y_test)
        
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
