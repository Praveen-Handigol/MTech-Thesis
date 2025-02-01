import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pandas as pd
import pickle
import torch
from bayes_opt import BayesianOptimization
import json
import time
import os

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="anom_detect.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_measures(actual, pred, tol):
    """
      A function to measure, the total True Positives, False Negatives and False Positives for a given tolerence.
      Parameters
      --------------
      actual : a 1-d array
         Actual labels (1 or 0 entries)
      pred : a 1-d array of same size of actual
         Predicted labels (predicted labels)
      tol : int
         tolerance value used for calculation

      Returns
      ---------------
      TP: int
        Total true positives count
      FN: int
         Total false negatives count
      FP: int
         Total false positives count
    """
    TP = 0
    FN = 0
    FP = 0
    actual = np.array(actual)
    pred = np.array(pred)
    if len(pred) != 0:
        for a in actual:
            if min(abs(pred - a)) <= tol:
                TP = TP + 1
            else:
                FN = FN + 1
        for p in pred:
            if min(abs(actual - p)) > tol:
                FP = FP + 1
    else:
        FN = len(actual)

    return TP, FN, FP


# Evaluation function
def evaluate(test_df, test_out_dict,window_size, min_height, tol=24, thresh=0.5, alpha=1, beta=0,
             only_peaks=False):
    """
    A function to evaluate the f1 scores on the test dataset.
    Parameters
    -----------
      test_df: pandas dataframe
        The dataset obtained for testing (processed and includes "s_no" column.
      test_out_dict: Dictionary
          Dictionary containing reconstruction details with s_no (segment number) as keys.
      window_size:
          size of the subsequence (number of datapoints)
      min_height: float in range (o,1)
         Setting threshold on the KDE curve
      thresh: float
        threshold on the reconstruction error to identify critical subsequences.
      alpha, beta: floats
         anomaly score params
      only_peaks: bool
        If true only the peaks in the region of KDE above min-height are marked as anomalous timestamps
        else, all the timestamps in those regions are marked as anomalies.
    """
    # error dict contains {s_no : [errors ]}
    TP = 0
    FN = 0
    FP = 0
    temp = test_df.groupby("s_no")
    for id, id_df in temp:
        id_df.reset_index(drop=True, inplace=True)
        id_dict = test_out_dict[id]
        error = id_dict["recon_loss"]
        z_norm = id_dict["Z"]
        if type(error) == torch.Tensor:
            error = error.detach().cpu().numpy()
        z_norm = torch.norm(z_norm.view(-1, lat_dim), dim=1).detach().cpu().numpy()
        combined_score = alpha * error + beta * z_norm
        mask = combined_score > thresh
        if not id_dict["window_b_included"]:
            mask = np.pad(mask, (window_size // 2 - 1,), mode='constant', constant_values=False)

        logger.info(f'{len(id_df)-len(mask),id_dict["window_a_included"],id_dict["window_b_included"]}')

        positions = np.where(mask)[0]
        if len(positions) <= 1:
            anom = positions
        else:
            kde = gaussian_kde(positions, bw_method=0.05)
            # Evaluate the KDE at some points
            x = range(0, len(id_df))
            y = kde(x)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

            if only_peaks:
                peaks, _ = find_peaks(y, height=min_height)
            else:
                peaks = np.where(y > min_height)[0]

            anom = peaks
        actual_anom = id_df.index[id_df['anomaly'] == 1]
        TP_i, FN_i, FP_i = get_measures(actual=actual_anom, pred=anom, tol=tol)
        TP = TP + TP_i
        FN = FN + FN_i
        FP = FP + FP_i
    return TP, FN, FP


if __name__ == "__main__":

    with open('config.json', 'r') as file:
        config = json.load(file)

    # select the  iters, window size and file
    window_size = config['preprocessing']['window_size']
    iters = config["recon"]["iters"]
    use_dtw = config["recon"]["use_dtw"]
    eval_mode = config["recon"]["use_eval_mode"]
    lat_dim = config['training']['latent_dim']

    tolerance = [12, 24]  # the tolerance values for which evaluation is required.


    # get the building ids
    df = pd.read_csv(config["data"]["dataset_path"])
    b_ids = df["building_id"].unique()
    del df
    logger.info(f"unique builds : {b_ids}")

    # b_ids = [1304] # or pass a custom list

    results_df = pd.DataFrame(
        columns=['b_id', 'use_dtw', 'alpha', 'beta', 'thresh', 'min_height', 'Precision', 'Recall', 'F1','tol'])
    for b_id in b_ids:
        logger.info(b_id)
        start_time = time.time()
        for dtw in [use_dtw]:  # also could check [True,False] if both are computed
          if os.path.exists(f"dataset/test_df_{b_id}.csv"):
            # Import the test files
            b_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")
            with open(f"test_out/iters_{iters}_reconstruction_{b_id}_{dtw}_{eval_mode}.pkl", "rb") as f:
                test_out_dict = pickle.load(f)

            # optimize the params:-

            def black_box_function(thresh, min_height, alpha, beta):
                "define the blackbox function to maximize with parameters as the variables to optimize"
                TP, FN, FP = evaluate(b_df, test_out_dict, window_size, min_height, 6, thresh, alpha, beta)
                logger.info(f"{TP, FN, FP}")
                try:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                    F1 = 2 * P * R / (P + R)
                except:
                    F1 = 0
                return F1


            # Bounded region of parameter space
            pbounds = {'thresh': (0, 100), 'min_height': (0.4, 1), "alpha": (0, 1), "beta": (0, 1)}

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=1, allow_duplicate_points=True
            )

            optimizer.maximize(
                init_points=70,
                n_iter=180
            )

            thresh = optimizer.max["params"]["thresh"]
            min_height = optimizer.max["params"]["min_height"]
            alpha = optimizer.max["params"]["alpha"]
            beta = optimizer.max["params"]["beta"]

            for tol in tolerance:
                TP, FN, FP = evaluate(b_df, test_out_dict, window_size, min_height, tol, thresh, alpha, beta)
                logger.info(f"{TP}, {FN}, {FP}")
                P = TP / (TP + FP)
                R = TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                logger.info(f"{P}, {R}, {F1}")
                results_df.loc[len(results_df)] = [b_id, dtw, alpha, beta, thresh, min_height, P, R, F1,tol]  # 'b_id',
                # 'use_dtw', 'alpha', 'beta', 'thresh', 'min_height', 'Precision', 'Recall', 'F1'
        end_time = time.time()
        logger.info(f"Time take for building {b_id} is {end_time-start_time}")
    if eval_mode:
        t1 = "eval_mode_on"
    else:
        t1 = "eval_mode_off"

    if use_dtw:
        t2 = "soft_dtw"
    else:
        t2 = "mse"

    results_df.to_csv(f"results_{t1}_{t2}.csv")
    logger.info(f"The result file is created: results_{t1}_{t2}.csv ")
