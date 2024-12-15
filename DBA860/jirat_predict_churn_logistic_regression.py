#*****************************************************************************
# Filename: jirat_predict_churn_logistic_regression.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For training a logistic regression model to predict churn
#*****************************************************************************

#-- Import Libraries ---------------------------------------------------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

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
        # Initialize the Logistic Regression model
        model = LogisticRegression(max_iter=1000)

        # Train the model on the training data
        model.fit(X_train, y_train)
        
        # Save the trained model to a file for future use
        model_filename = config.MODELS_ROOT + config.LOGISTIC_REGRESSION_MODEL
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
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Print coefficients to interpret feature influence
        coefficients = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })
        print()
        print(f"*******************************************************")
        print(coefficients.sort_values(by="Coefficient", ascending=False))
        print(f"*******************************************************")

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # Print evaluation metrics
        print()
        print(f"****************************************")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
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
        # Get predicted probabilities for the test data
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Visualization: ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color="blue", label="ROC Curve")
        plt.plot([0, 1], [0, 1], "--r")  # Line for no-skill classifier
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
        
        # Visualization: Coefficients
        coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
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
        print(f"Churn Prediction - Model Training")
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
