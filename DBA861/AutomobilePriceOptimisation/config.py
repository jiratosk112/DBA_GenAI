DATA_FILE = f"/projects/ggu/DBA_GenAI/DBA861/AutomobilePriceOptimisation/data/Automobile price data - Raw.csv"
MODELS_ROOT = f"/projects/GGU/DBA_GENAI/DBA861/AutomobilePriceOptimisation/models/"

FNN = f"automobile_price_prediction_fnn.h5.keras"
REGRESSION = f"automobile_price_prediction_regression.h5.keras"

X_TEST_FNN = "X_test_fnn.pkl"
Y_TEST_FNN = "Y_test_fnn.pkl"
PREPROCESSOR_FNN = "preprocessor_fnn.pkl"

X_TEST_REGRESSION = "X_test_regression.pkl"
Y_TEST_REGRESSION = "Y_test_regression.pkl"
SCALER_REGRESSION = "scaler_regression.pkl"

# Define new car features for prediction
new_car_features = {
    'symboling': 3,
    'normalized-losses': 100,
    'make': 'audi',
    'fuel-type': 'gas',
    'aspiration': 'std',
    'num-of-doors': 'four',
    'body-style': 'sedan',
    'drive-wheels': 'fwd',
    'engine-location': 'front',
    'wheel-base': 99.4,
    'length': 176.6,
    'width': 66.2,
    'height': 54.3,
    'curb-weight': 2337,
    'engine-type': 'ohc',
    'num-of-cylinders': 'four',
    'engine-size': 109,
    'fuel-system': 'mpfi',
    'bore': 3.19,
    'stroke': 3.4,
    'compression-ratio': 8.0,
    'horsepower': 102,
    'peak-rpm': 5500,
    'city-mpg': 24,
    'highway-mpg': 30
}