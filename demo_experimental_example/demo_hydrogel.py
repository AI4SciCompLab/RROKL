"""
This script runs the hydrogel constitutive model example for the paper:
"A robust thermodynamically consistent kernel learning framework for knowledge-intensive
 discovery of path-dependent constitutive models" published in Advanced Engineering Informatics.

It loads pre-trained RROKL model parameters from a pickle file, trains the model on
different ratios of training data with outliers, and evaluates interpolation
performance on a clean test dataset. Results are saved to a .npz file.
"""

import pickle
import time
import numpy as np
from scipy.io import loadmat
import os
from rrokl import RROKL   # Ensure the above class is imported

# ---------- Set paths ----------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
open_dir = "hydrogel"

# ---------- Parameters ----------
ratios = np.arange(0, 0.5, 0.1)   
n_ratios = len(ratios)
n_simulations = 5

R2 = np.zeros((n_simulations, n_ratios))
NRMSE = np.zeros((n_simulations, n_ratios))

test_filename = os.path.join(open_dir, f'hydrogel_clean.mat')
test_data = loadmat(test_filename)
test_set = test_data['hydrogel_clean']  
strain_test = test_set[:, 0]
stress_test = test_set[:, 1]

for j in range(1, n_simulations+1):
    # Load saved parameter dictionary
    with open(os.path.join(open_dir, f'RROKL_Architecture_Hydrogel{j}_all.pkl'), 'rb') as f:
        all_params = pickle.load(f)

    for i, ratio in enumerate(ratios):

        # 1. Load pre-trained model parameters from saved dictionary
        key = f'ratio_{int(ratio*100)}'
        params = all_params[key]    
        opt_gamma = params['gamma'].item()      
        opt_sigma2 = params['sigma2'].item()            
        opt_Xsub = params['Xsub']        

        # 2. Load training data
        train_filename = os.path.join(open_dir, f'hydrogel_outlier_set{j}-{int(ratio*100)}.mat')
        train_data = loadmat(train_filename)
        train_set = train_data['hydrogel_outlier']  
        strain_train = train_set[:, 0]
        stress_train = train_set[:, 1]

        # 3. Compute thresholds and expand features
        numR = 5
        r = RROKL.compute_thresholds(strain_train, numR)   
        operator = "stop"

        Xtr = RROKL.expand_features(strain_train, r, operator)
        Ytr = stress_train

        # 4. Initialize model and set pre-trained parameters
        model = RROKL(gamma=opt_gamma, sigma2=opt_sigma2,
                    num_basis=opt_Xsub.shape[0],
                    preprocess='on', weight_type='Logistic',
                    patience_count=5, tol=1e-8,
                    num_operators=numR, operator_type=operator,
                    thresholds=r)   # Save thresholds for prediction use
        model.set_basis(opt_Xsub)     # Use pre-selected basis vectors
        start = time.perf_counter()
        model.fit(Xtr, Ytr)           # Training
        end = time.perf_counter()
        print(f"Training time for ratio {ratio:.1f}: {end - start:.2f} seconds")

        # Prediction (model has stored thresholds, auto-expand)
        yp_test = model.predict(strain_test, expand=True, scale=True)
        R2[j-1, i] = model.score(strain_test, stress_test, metric='R2')
        NRMSE[j-1, i] = model.score(strain_test, stress_test, metric='NRMSE')
        print(f"The {j}-th set {i+1}-th ratio ({ratio:.1f}) is finished!")

# Optional: Save results
np.savez('results_hydrogel.npz', R2=R2, NRMSE=NRMSE)