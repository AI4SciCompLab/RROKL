"""
This script runs the uniaxial elastoplasticity example for the paper:
"A robust thermodynamically consistent kernel learning framework for knowledge-intensive
 discovery of path-dependent constitutive models" published in the Advanced Engineering Informatics.

It loads pre-trained RROKL model parameters from a pickle file, trains the model on
different outlier corruption ratios of training data, and evaluates interpolation and extrapolation
performance across multiple simulated test datasets. Results are saved to a .npz file.
"""

import pickle
import time
import numpy as np
from scipy.io import loadmat
import os
from rrokl import RROKL   # Ensure the above class is imported

# ---------- Set paths ----------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
open_dir1 = "Training_data"
open_dir2 = "Test_data_interpolation"
open_dir3 = "Test_data_extrapolation"

# Load saved parameter dictionary
with open('RROKL_Architecture_Case_all.pkl', 'rb') as f:
    all_params = pickle.load(f)

# ---------- Parameters ----------
ratios = np.arange(0, 0.5, 0.1)   # 0,0.1,0.2,0.3,0.4
n_ratios = len(ratios)
n_simulations = 20

R2_extra = np.zeros((n_simulations, n_ratios))
NRMSE_extra = np.zeros((n_simulations, n_ratios))
R2_int = np.zeros((n_simulations, n_ratios))
NRMSE_int = np.zeros((n_simulations, n_ratios))

for i, ratio in enumerate(ratios):

    # 1. Load pre-trained model parameters from saved dictionary
    key = f'ratio_{int(ratio*100)}'
    params = all_params[key]    
    opt_gamma = params['gamma'].item()      # float
    opt_sigma2 = params['sigma2'].item()    # float        
    opt_Xsub = params['Xsub']        # NumPy 2D array

    # 2. Load training data
    train_filename = os.path.join(open_dir1, f'train_set{int(ratio*100)}.mat')
    train_data = loadmat(train_filename)
    train_set1 = train_data['train_path1']  
    strain1 = train_set1[:, 0]
    stress1 = train_set1[:, 1]

    train_set2 = train_data['train_path2']  
    strain2 = train_set2[:, 0]
    stress2 = train_set2[:, 1]

    train_set3 = train_data['train_path3']  
    strain3 = train_set3[:, 0]
    stress3 = train_set3[:, 1]

    # 3. Compute thresholds and expand features
    strain = np.concatenate((strain1, strain2, strain3))  
    numR = 50
    r = RROKL.compute_thresholds(strain, numR)   
    operator = "stop"

    X1 = RROKL.expand_features(strain1, r, operator)
    X2 = RROKL.expand_features(strain2, r, operator)
    X3 = RROKL.expand_features(strain3, r, operator)
    Xtr = np.vstack((X1, X2, X3))
    Ytr = np.hstack((stress1, stress2, stress3))

    # 4. Initialize model and set pre-trained parameters
    model = RROKL(gamma=opt_gamma, sigma2=opt_sigma2,
                  num_basis=opt_Xsub.shape[0],
                  preprocess='on', weight_type='Logistic',
                  patience_count=5, tol=1e-6,
                  num_operators=numR, operator_type=operator,
                  thresholds=r)   # Save thresholds for prediction use
    model.set_basis(opt_Xsub)     # Use pre-selected basis vectors
    start = time.perf_counter()
    model.fit(Xtr, Ytr)           # Training
    end = time.perf_counter()
    print(f"Training time for ratio {ratio:.1f}: {end - start:.2f} seconds")

    # 5. Test interpolation and extrapolation on each simulated dataset
    for j in range(1, n_simulations+1):
        # Interpolation test
        int_filename = os.path.join(open_dir2, f'test_set{j}.mat')
        int_data = loadmat(int_filename)
        test_set = int_data['test_set']
        strain_test = test_set[:, 0]
        stress_test = test_set[:, 1]

        # Prediction (model has stored thresholds, auto-expand)
        yp_test = model.predict(strain_test, expand=True, scale=True)
        R2_int[j-1, i] = model.score(strain_test, stress_test, metric='R2')
        NRMSE_int[j-1, i] = model.score(strain_test, stress_test, metric='NRMSE')

        # Extrapolation test
        extra_filename = os.path.join(open_dir3, f'test_set{j}.mat')
        extra_data = loadmat(extra_filename)
        test_set = extra_data['test_set']
        strain_test = test_set[:, 0]
        stress_test = test_set[:, 1]

        yp_test = model.predict(strain_test, expand=True, scale=True)
        R2_extra[j-1, i] = model.score(strain_test, stress_test, metric='R2')
        NRMSE_extra[j-1, i] = model.score(strain_test, stress_test, metric='NRMSE')

    print(f"The {i+1}-th ratio ({ratio:.1f}) is finished!")

# Save results
np.savez('results_uniaxial_case.npz', R2_int=R2_int, NRMSE_int=NRMSE_int,
         R2_extra=R2_extra, NRMSE_extra=NRMSE_extra)