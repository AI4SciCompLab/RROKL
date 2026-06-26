"""
This script runs the multiaxial elastoplasticity example for the paper:
"A robust thermodynamically consistent kernel learning framework for knowledge-intensive
 discovery of path-dependent constitutive models" published in Advanced Engineering Informatics.

It loads pre-trained RROKL3D model parameters, trains the model on different ratios of
training data, and evaluates interpolation performance across multiple simulated test datasets.
Results are saved to a .npz file.
"""

import pickle
import numpy as np
from scipy.io import loadmat
import os
from rrokl3d import RROKL3D

# ---------- Path settings ----------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
open_dir1 = "Training_data"   
open_dir2 = "Test_data"     

# Load saved parameter dictionary
with open('RROKL_Architecture_Case_all.pkl', 'rb') as f:
    all_params = pickle.load(f)

# ---------- Parameters ----------
ratios = np.arange(0, 0.5, 0.1)        
n_ratios = len(ratios)
n_simulations = 20

# Store results (R2/NRMSE for each output per test set, and mean)
R2 = np.zeros((n_simulations, n_ratios, 4))   
NRMSE = np.zeros((n_simulations, n_ratios, 4))
R2_mean = np.zeros((n_simulations, n_ratios))
NRMSE_mean = np.zeros((n_simulations, n_ratios))

# ---------- Main loop ----------
for i, ratio in enumerate(ratios):
    ratio_str = f"{int(ratio*100):03d}"
    print(f"\n=== Processing ratio {ratio:.1f} ({ratio_str}) ===")

    # 1. Load pre-trained model parameters from saved dictionary
    key = f'ratio_{int(ratio*100)}'
    params = all_params[key]    
    opt_gamma = params['gamma'].item()     
    opt_sigma2 = params['sigma2'].item()           
    opt_Xsub = params['Xsub']       

    # 2. Load training data
    train_filename = os.path.join(open_dir1, f'train_set{int(ratio*100)}.mat')
    train_data = loadmat(train_filename)
    train_set1 = train_data['train_path1']  
    strain1 = train_set1[:, :3]
    stress1 = train_set1[:, 3:]

    train_set2 = train_data['train_path2']  
    strain2 = train_set2[:, :3]
    stress2 = train_set2[:, 3:]

    train_set3 = train_data['train_path3']  
    strain3 = train_set3[:, :3]
    stress3 = train_set3[:, 3:]

    # 3. Compute principal directions and back stress (using all strain)
    strain = np.concatenate((strain1, strain2, strain3))  
    principal_dirs, back_stress_train = RROKL3D.compute_principal_directions(strain)
    n1 = strain1.shape[0]
    n2 = strain2.shape[0]

    # 4. Compute thresholds for each back stress component
    numR = 30
    K = back_stress_train.shape[1]   
    thresholds_list = []
    for k in range(K):
        rk = RROKL3D.compute_thresholds(back_stress_train[:, k], numR)
        thresholds_list.append(rk)

    # 5. Build features for each segment (original strain + expanded back stress)
    def build_segment_features(strain_seg, back_stress_seg):
        # First compute back stress for the segment (projection)
        back_seg = RROKL3D.project_strain(strain_seg, principal_dirs)
        # Build features
        X_seg = RROKL3D.build_features(strain_seg, back_seg, thresholds_list, operator_type='stop')
        return X_seg, back_seg

    X1, _ = build_segment_features(strain1, back_stress_train[:n1])
    X2, _ = build_segment_features(strain2, back_stress_train[n1:n1+n2])
    X3, _ = build_segment_features(strain3, back_stress_train[n1+n2:])

    # Concatenate features and targets
    Xtr = np.vstack((X1, X2, X3))
    Ytr = np.vstack((stress1, stress2, stress3))

    # 7. Initialize model
    model = RROKL3D(gamma=opt_gamma, sigma2=opt_sigma2,
                    num_basis=opt_Xsub.shape[0],
                    preprocess='on',
                    weight_type='Logistic',
                    patience_count=8,
                    tol=1e-6,
                    num_operators=numR,
                    operator_type='stop',
                    thresholds=thresholds_list)   # Store thresholds for prediction
    model.set_basis(opt_Xsub)
    model.fit(Xtr, Ytr)

    # 8. Predict on each interpolation test set
    for j in range(1, n_simulations+1):
        test_file = os.path.join(open_dir2, f"test_set{j}.mat")
        test_data = loadmat(test_file)
        test_set = test_data['test_set']   
        strain_test = test_set[:, :3]
        stress_true = test_set[:, 3:]

        # Compute test set back stress (projection)
        back_stress_test = RROKL3D.project_strain(strain_test, principal_dirs)

        # Build test features
        Xt = RROKL3D.build_features(strain_test, back_stress_test, thresholds_list, operator_type='stop')

        # Predict
        yp_test = model.predict(Xt, expand=False, scale=True)   

        # Compute R2 and NRMSE for each output
        r2_each = model.score(Xt, stress_true, metric='R2', expand=False, scale=True)
        nrmse_each = model.score(Xt, stress_true, metric='NRMSE', expand=False, scale=True)
        R2[j-1, i, :] = r2_each
        NRMSE[j-1, i, :] = nrmse_each
        R2_mean[j-1, i] = np.mean(r2_each)
        NRMSE_mean[j-1, i] = np.mean(nrmse_each)

    print(f"Finished ratio {ratio:.1f} ({i+1}/{n_ratios})")

# Save results
np.savez('results_multiaxial_case.npz',
         R2=R2, NRMSE=NRMSE, R2_mean=R2_mean,
         NRMSE_mean=NRMSE_mean)
print("\nAll done. Results saved to results_multiaxial.npz")