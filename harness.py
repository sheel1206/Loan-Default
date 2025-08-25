import argparse
import pandas as pd
import numpy as np
from preprocess import load_data, preprocess_data, feature_selection
import pickle
import statsmodels.api as sm
from prediction import prediction_harness

# Takes the input and outout CSV arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="Path to the input CSV file.")
parser.add_argument("--output_csv", type=str, help="Path to the output CSV file.")
args = parser.parse_args()
# Loads the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

print("Loading input data...")
data = preprocess_data(args.input_csv) # gives the features to be used for the chosen model
X = feature_selection(data)
#X_w_const = sm.add_constant(X, has_constant='add')
print("Running inference...")
# Perform prediction as well as Calibration
#logit_params = np.array([-0.20619009, -6.39329043,  0.20819009])
#gb_params = np.array([-0.25477479, -4.46701662,  0.25677479])
gb_params = np.array([-0.26852134, -3.99542877, 0.26840296])
predictions = prediction_harness(X, model, calibration_params = gb_params)
preds = pd.Series(predictions)
#preds = model.predict_proba(X)[:, 1]
#preds = pd.Series(preds)
#calibrated predictions to csv
preds.to_csv(args.output_csv, index=False, header = False)
print(f"Predictions saved to {args.output_csv}")

print("Harness Complete.")