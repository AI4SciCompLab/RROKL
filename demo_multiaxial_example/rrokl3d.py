import numpy as np
from scipy.linalg import solve
from sklearn.utils import check_random_state

class RROKL3D:
    """
    Robust Reduced-Order Kernel Learning for 3D (multi-output) problems.
    Supports multiple output dimensions (e.g., back stress components).
    """

    def __init__(self, gamma=1.0, sigma2=1.0, num_basis=100,
                 preprocess='on', weight_type='Huber',
                 patience_count=20, tol=1e-6,
                 num_operators=10, operator_type='stop',
                 random_state=None, thresholds=None):
        """
        Parameters
        ----------
        gamma : float, regularization parameter (shared for all outputs)
        sigma2 : float, RBF kernel bandwidth squared
        num_basis : int, number of basis vectors (ignored if Xsub is set)
        preprocess : 'on' or 'off', standardize features and targets
        weight_type : str, weight function type
        patience_count : int, early stopping patience
        tol : float, convergence tolerance
        num_operators : int, number of hysteresis thresholds
        operator_type : str, 'play', 'stop', 'dead-zone', 'tangent'
        random_state : int or None, seed for basis selection
        thresholds : array or None, precomputed thresholds
        """
        self.gamma = gamma
        self.sigma2 = sigma2
        self.num_basis = num_basis
        self.preprocess = preprocess
        self.weight_type = weight_type
        self.patience_count = patience_count
        self.tol = tol
        self.num_operators = num_operators
        self.operator_type = operator_type
        self.random_state = random_state
        self.thresholds = thresholds

        # Model parameters (will be lists for multi-output)
        self.Xsub = None               # shared basis (scaled)
        self.model_pars = []           # list of (beta, bias) for each output
        self.scaled_Xtr = None
        self.scaled_ytr = None         # matrix, rows=samples, cols=outputs
        self.mean_Xtr = None
        self.std_Xtr = None
        self.mean_ytr = None           # array of length n_outputs
        self.std_ytr = None            # array of length n_outputs

    # ---------- Static helper methods (same as before) ----------
    @staticmethod
    def compute_thresholds(strain, num_operators):
        max_strain = np.max(np.abs(strain))
        return np.array([(i+1)*max_strain/(num_operators+1) for i in range(num_operators)])

    @staticmethod
    def expand_features(x, r, operator_type):
        x = np.asarray(x, dtype=np.float64).ravel()
        r = np.asarray(r, dtype=np.float64).ravel()
        N = len(x); M = len(r)
        X = np.zeros((N, M+1), dtype=np.float64)
        X[:, 0] = x
        Pr_t = np.zeros((N, M), dtype=np.float64)

        for i in range(M):
            for j in range(N):
                if operator_type == 'play':
                    if j == 0:
                        Pr_t0 = np.maximum(x[j] - r[i], np.minimum(x[j] + r[i], 0.0))
                        Pr_t[j, i] = Pr_t0
                    else:
                        Pr_t[j, i] = np.maximum(x[j] - r[i],
                                                np.minimum(x[j] + r[i], Pr_t[j-1, i]))
                elif operator_type in ('stop', 'dead-zone', 'tangent'):
                    if j == 0:
                        Pr_t0 = np.minimum(r[i], np.maximum(-r[i], x[j]))
                        Pr_t[j, i] = Pr_t0
                    else:
                        if operator_type == 'stop':
                            prev = x[j-1]
                        else:
                            prev = x[0]
                        Pr_t[j, i] = np.minimum(r[i],
                                                np.maximum(-r[i],
                                                           x[j] - prev + Pr_t[j-1, i]))
                else:
                    raise ValueError(f"Unsupported operator type: {operator_type}")
        X[:, 1:] = Pr_t
        return X

    @staticmethod
    def _kernelmatrix(Xtrain, sigma2, Xtest=None):
        sigma2 = float(sigma2)
        Xtrain = np.asarray(Xtrain, dtype=np.float64)
        if Xtest is not None:
            Xtest = np.asarray(Xtest, dtype=np.float64)

        if Xtest is None:
            sq_norm = np.sum(Xtrain ** 2, axis=1, keepdims=True)
            K = sq_norm + sq_norm.T - 2 * Xtrain @ Xtrain.T
        else:
            if Xtrain.shape[1] != Xtest.shape[1]:
                raise ValueError("Feature dimensions mismatch.")
            sq_norm_train = np.sum(Xtrain ** 2, axis=1, keepdims=True).T
            sq_norm_test = np.sum(Xtest ** 2, axis=1, keepdims=True)
            K = sq_norm_test + sq_norm_train - 2 * Xtest @ Xtrain.T
        return np.exp(-K / (2 * sigma2))

    @staticmethod
    def _weight_function(residual, weight_type):
        e = np.asarray(residual, dtype=np.float64).ravel()
        n = len(e)
        Beta = np.zeros(n, dtype=np.float64)
        s_hat = 1.483 * np.median(np.abs(e - np.median(e)))
        r = e / (s_hat + 1e-12)

        if weight_type == 'Huber':
            for i in range(n):
                Beta[i] = 1.0 if np.abs(r[i]) < 1.345 else 1.345 / np.abs(r[i])
        elif weight_type == 'Hampel':
            for i in range(n):
                absr = np.abs(r[i])
                if absr < 2.5:
                    Beta[i] = 1.0
                elif absr <= 3.0:
                    Beta[i] = (3.0 - absr) / 0.5
                else:
                    Beta[i] = 1e-6
        elif weight_type == 'Logistic':
            for i in range(n):
                Beta[i] = np.tanh(r[i]) / (r[i] + 1e-12)
        elif weight_type == 'Myriad':
            Q1 = np.percentile(e, 25)
            Q3 = np.percentile(e, 75)
            delta = 0.5 * (Q3 - Q1)
            for i in range(n):
                Beta[i] = delta ** 2 / (delta ** 2 + r[i] ** 2 + 1e-12)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        return Beta

    # ---------- PCA for multi-axial strain ----------
    @staticmethod
    def compute_principal_directions(strain_data):
        """
        Compute principal directions from strain data (PCA on 6D strain space).
        
        Parameters
        ----------
        strain_data : array, shape (n_samples, 3)
            Columns: epsilon11, epsilon22, epsilon12 (engineering shear)
        
        Returns
        -------
        principal_directions : array, shape (6, K)
            Normalized principal directions (eigenvectors with non-zero eigenvalues)
        back_stress : array, shape (n_samples, K)
            Projection of strain onto principal directions
        """
        strain11 = strain_data[:, 0]
        strain22 = strain_data[:, 1]
        strain12 = strain_data[:, 2]
        n = len(strain11)
        strain6 = np.zeros((n, 6), dtype=np.float64)
        strain6[:, 0] = strain11
        strain6[:, 1] = strain22
        strain6[:, 3] = strain12   # engineering shear to tensor component
        
        # Center data
        mean_strain = np.mean(strain6, axis=0)
        centered = strain6 - mean_strain
        
        # Covariance matrix
        cov_matrix = np.cov(centered, rowvar=False)
        
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
        # eigh returns ascending, we want > tolerance
        tol = 1e-10
        mask = eigen_vals > tol
        principal_directions = eigen_vecs[:, mask]
        K = principal_directions.shape[1]
        
        # Normalize
        for k in range(K):
            principal_directions[:, k] /= np.linalg.norm(principal_directions[:, k])
        
        # Project
        back_stress = strain6 @ principal_directions
        return principal_directions, back_stress

    @staticmethod
    def project_strain(strain_data, principal_directions):
        """
        Project strain data onto given principal directions.
        """
        strain11 = strain_data[:, 0]
        strain22 = strain_data[:, 1]
        strain12 = strain_data[:, 2]
        n = len(strain11)
        strain6 = np.zeros((n, 6), dtype=np.float64)
        strain6[:, 0] = strain11
        strain6[:, 1] = strain22
        strain6[:, 3] = strain12
        return strain6 @ principal_directions
    
    @staticmethod
    def build_features(strain, back_stress, thresholds_list, operator_type='stop'):
        """
        Build feature matrix from strain and back stress components.
        
        Parameters
        ----------
        strain : array, shape (n_samples, 3)  [eps11, eps22, eps12]
        back_stress : array, shape (n_samples, K)  each column is one back stress component
        thresholds_list : list of arrays, length K, each array of thresholds for that component
        operator_type : str, hysteresis operator type
        
        Returns
        -------
        X : array, shape (n_samples, 3 + K*(numR+1))
        """
        n = strain.shape[0]
        K = back_stress.shape[1]
        # 先复制原始应变
        X = [strain]
        for k in range(K):
            r = thresholds_list[k]
            Xk = RROKL3D.expand_features(back_stress[:, k], r, operator_type)
            X.append(Xk)
        return np.hstack(X)

    # ---------- Basis setting ----------
    def set_basis(self, Xsub):
        self.Xsub = np.asarray(Xsub, dtype=np.float64)
        self.num_basis = self.Xsub.shape[0]

    # ---------- Training ----------
    def fit(self, X, Y):
        """
        Train the model with multi-output targets using joint convergence criterion.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same number of samples")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_outputs = Y.shape[1]

        # Preprocess
        if self.preprocess == 'on':
            self.mean_Xtr = np.mean(X, axis=0)
            self.std_Xtr = np.std(X, axis=0, ddof=0)
            self.std_Xtr[self.std_Xtr == 0] = 1.0
            X_scaled = (X - self.mean_Xtr) / self.std_Xtr

            self.mean_ytr = np.mean(Y, axis=0)
            self.std_ytr = np.std(Y, axis=0, ddof=0)
            self.std_ytr[self.std_ytr == 0] = 1.0
            Y_scaled = (Y - self.mean_ytr) / self.std_ytr
        else:
            self.mean_Xtr = None
            self.std_Xtr = None
            self.mean_ytr = None
            self.std_ytr = None
            X_scaled = X
            Y_scaled = Y

        self.scaled_Xtr = X_scaled
        self.scaled_ytr = Y_scaled

        # Select basis if not set
        if self.Xsub is None:
            rng = check_random_state(self.random_state)
            idx = rng.permutation(X_scaled.shape[0])[:self.num_basis]
            self.Xsub = X_scaled[idx, :]
        else:
            if self.Xsub.shape[1] != X_scaled.shape[1]:
                raise ValueError("Xsub dimension mismatch with training data.")

        # Precompute kernel matrices
        K = self._kernelmatrix(self.Xsub, self.sigma2)          # (nb, nb)
        Ke = self._kernelmatrix(self.Xsub, self.sigma2, X_scaled).T  # (nb, n)
        nb = K.shape[0]
        KA = np.zeros((nb + 1, nb + 1), dtype=np.float64)
        KA[:nb, :nb] = K
        KI = np.vstack((Ke, np.ones((1, Ke.shape[1]), dtype=np.float64)))
        # A_const = (KA + KA.T) / (2.0 * self.gamma)
        A_const = (KA + KA.T)*self.gamma / 2.0

        # Initialize model parameters (list of arrays)
        self.model_pars = [np.zeros(nb + 1) for _ in range(n_outputs)]
        # Initialize weight matrices (identity for each output)
        B_list = [np.eye(X_scaled.shape[0], dtype=np.float64) for _ in range(n_outputs)]

        residual_old = np.zeros((X_scaled.shape[0], n_outputs), dtype=np.float64)
        cond_best = np.inf
        count = 0
        max_iter = 500
        tol = self.tol
        patience = self.patience_count

        for it in range(max_iter):
            # 1. Solve for each output using current B_list
            for out_idx in range(n_outputs):
                B = B_list[out_idx]
                y = Y_scaled[:, out_idx]
                A = A_const + KI @ B @ KI.T
                b = KI @ B @ y
                pars = solve(A, b, assume_a='sym')
                self.model_pars[out_idx] = pars

            # 2. Compute joint predictions and residuals
            Ypred_scaled = self.predict(self.scaled_Xtr, expand=False, scale=False)
            residual_new = Ypred_scaled - Y_scaled  # (n_samples, n_outputs)

            # 3. Joint convergence check
            cond1 = np.linalg.norm(residual_new - residual_old, 'fro')
            cond2 = np.mean(residual_new ** 2)

            if cond1 < tol or cond2 < tol:
                print(f"Convergence reached after {it+1} iteration(s)")
                break

            if cond1 <= cond_best:
                cond_best = cond1
            else:
                count += 1
            if count > patience:
                print(f"Early stopping after {it+1} iteration(s)")
                break

            # 4. Update residuals and weights for next iteration
            residual_old = residual_new
            for out_idx in range(n_outputs):
                e = residual_old[:, out_idx]
                Beta = self._weight_function(e, self.weight_type)
                B_list[out_idx] = np.diag(Beta)

            if it == max_iter - 1:
                print(f"Warning: Max iterations reached, f(x) = {cond1:.6f}, residual norm = {cond2:.6f}")

        # Store final model parameters
        self.model_pars = self.model_pars

    # ---------- Prediction ----------
    def predict(self, X, expand=True, scale=True):
        """
        Predict multiple outputs.
        
        Parameters
        ----------
        X : array, shape (n_samples,) or (n_samples, n_features)
            If expand=True, X is raw scalar signal; else expanded features.
        expand : bool, whether to apply hysteresis expansion
        scale : bool, whether to apply standardization
        
        Returns
        -------
        Ypred : array, shape (n_samples, n_outputs)
        """
        if expand:
            if self.thresholds is None:
                raise ValueError("Thresholds not set for expansion.")
            X_exp = self.expand_features(X, self.thresholds, self.operator_type)
        else:
            X_exp = np.asarray(X, dtype=np.float64)

        if scale and self.preprocess == 'on':
            X_scaled = (X_exp - self.mean_Xtr) / self.std_Xtr
        else:
            X_scaled = X_exp

        Kt = self._kernelmatrix(self.Xsub, self.sigma2, X_scaled)  # (n_test, nb)
        n_outputs = len(self.model_pars)
        Ypred_scaled = np.zeros((X_scaled.shape[0], n_outputs), dtype=np.float64)
        for i, pars in enumerate(self.model_pars):
            beta = pars[:-1]
            bias = pars[-1]
            Ypred_scaled[:, i] = Kt @ beta + bias

        if scale and self.preprocess == 'on':
            return Ypred_scaled * self.std_ytr + self.mean_ytr
        else:
            return Ypred_scaled

    # ---------- Scoring ----------
    def score(self, X, Y_true, metric='R2', expand=True, scale=True):
        """
        Compute R² or NRMSE for multi-output.
        Returns array of metrics for each output, or average if requested.
        """
        Ypred = self.predict(X, expand=expand, scale=scale)
        Y_true = np.asarray(Y_true, dtype=np.float64)
        if Y_true.ndim == 1:
            Y_true = Y_true.reshape(-1, 1)
        n_outputs = Y_true.shape[1]
        scores = []
        for i in range(n_outputs):
            y = Y_true[:, i]
            yp = Ypred[:, i]
            if metric.upper() == 'R2':
                sse = np.sum((y - yp) ** 2)
                sst = np.sum((y - np.mean(y)) ** 2)
                scores.append(1 - sse / sst)
            elif metric.upper() == 'NRMSE':
                rmse = np.sqrt(np.mean((y - yp) ** 2))
                scores.append(rmse / (np.max(y) - np.min(y)))
            else:
                raise ValueError("Metric must be 'R2' or 'NRMSE'")
        return np.array(scores)