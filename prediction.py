import numpy as np
import pandas as pd


"""
Function to flatten ground truth and predictions lists across years
"""

def flatten_pred(preds_probs):
    if isinstance(preds_probs, float):
        preds_flattened = [preds_probs]  # Wrap it in a list
    else:
        preds_flattened = [item for sublist in preds_probs for item in sublist]
  # return lists
    return preds_flattened

"""
Function which implements the exponential function with calibration params
"""

def calibration_function(params, x):

  a = float(params[0])
  b = float(params[1])
  c = float(params[2])
  y = a * np.exp(b * x) + c

  return y

"""
Function which adjusts calibrated parameters so p > 0 + epsilon
"""

def adjust_calibration_params(params, epsilon=0.002):

  a, b, c = params

  # at x = 0, y = a + c
  if a + c < epsilon:
    c = epsilon - a
  new_params = np.array([a, b, c])

  return new_params

"""
Function which maps predicted probabilities to calibrated probabilities
"""

def obtain_calibrated_probs(preds_probs, params):

  calibrated_probs = []

  # map each predicted prob to calibrated prob
  for i in preds_probs:
    #original_prob = i
    prob = calibration_function(params, i)
    calibrated_probs.append(prob)

  return calibrated_probs

"""
Function which maps predicted probs to calibrated probs for each year
"""

def obtain_calibrated_probs_years(preds_probs, params):

  cal_probs_years = []

  # iterate through each year
  for i in range(len(preds_probs)):
    curr_probs = [j for j in preds_probs[i]]

    # calibrate probabilities
    cal_probs = obtain_calibrated_probs(curr_probs, params)
    cal_probs_years.append(cal_probs)

  return cal_probs_years

"""
PREDICTION HARNESS: takes in a preprocessed feature set, outputs calibrated predictions
"""

def prediction_harness(preprocessed_feature_set, model , calibration_params, flattened=True, ):

  # step 1: use walk-forward analysis to train & predict
  #preds_probs = model.predict(preprocessed_feature_set)
  preds_probs = model.predict_proba(preprocessed_feature_set)[:, 1]

# step 3: flatten labels & predictions across years
#   preds_flattened = flatten_pred(preds_probs)

#   # step 4: plot calibration graph & obtain function parameters
#   calibration_params = plot_calibration_graph(preds_flattened, ground_truth_flattened)
#   'a, b, c values of a * e^(bx) + c: [-0.33817705 -3.58627177  0.33640614]'
#   'new parameters after feature selection: array([-0.20619009, -6.39329043,  0.20819009])'

  # step 5: adjust calibration parameters so that p > 0
  calibration_params = adjust_calibration_params(calibration_params)

  # step 6: map flattened probabilities to calibrated probabilities
  calibrated_probs = obtain_calibrated_probs(preds_probs, calibration_params)

  return calibrated_probs

