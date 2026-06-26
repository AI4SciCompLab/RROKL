import numpy as np
from scipy.linalg import solve, pinv
from sklearn.utils import check_random_state

class RROKL:
    """
    Robust Reduced Order Kernel Learning (RROKL)
    """

    def __init__(self, gamma=1.0, sigma2=1.0, num_basis=100,
                 preprocess='on', weight_type='Huber',
                 patience_count=20, tol=1e-6,
                 num_operators=10, operator_type='play',
                 random_state=None, thresholds=None):
        """
        Parameters
        ----------
        gamma : float, regularization parameter
        sigma2 : float, kernel bandwidth squared
        num_basis : int, number of basis vectors (ignored if Xsub is set)
        preprocess : 'on' or 'off', whether to standardize data
        weight_type : str, 'Huber', 'Hampel', 'Logistic', 'Myriad'
        patience_count : int, early stopping patience
        tol : float, convergence tolerance
        num_operators : int, number of hysteresis operators (thresholds)
        operator_type : str, 'play', 'stop', 'dead-zone', 'tangent'
        random_state : int or None
        thresholds : array or None, precomputed thresholds (if None, computed in fit)
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
        self.thresholds = thresholds   # can be set later

        # placeholders
        self.Xsub = None               # basis vectors (scaled)
        self.model_pars = None         # beta + bias
        self.scaled_Xtr = None
        self.scaled_ytr = None
        self.mean_Xtr = None
        self.std_Xtr = None
        self.mean_ytr = None
        self.std_ytr = None

    # ------------------ Static helper methods ------------------

    @staticmethod
    def compute_thresholds(strain, num_operators):
        """Generate threshold vector for hysteresis operators."""
        max_strain = np.max(np.abs(strain))
        r = np.zeros(num_operators)
        for i in range(num_operators):
            r[i] = (i + 1) * max_strain / (num_operators + 1)
        return r

    @staticmethod
    def expand_features(x, r, operator_type):
        """
        Apply hysteresis operators to input signal x.

        Parameters
        ----------
        x : 1D array, input signal
        r : 1D array, thresholds
        operator_type : str, 'play', 'stop', 'dead-zone', 'tangent'

        Returns
        -------
        X : 2D array, shape (len(x), len(r)+1)
        """
        x = np.asarray(x).ravel()
        r = np.asarray(r).ravel()
        N = len(x)
        M = len(r)
        X = np.zeros((N, M + 1))
        X[:, 0] = x
        Pr_t = np.zeros((N, M))

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
                        else:  # dead-zone or tangent
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
        """RBF kernel matrix."""
        Xtrain = np.asarray(Xtrain)
        if Xtest is None:
            sq_norm = np.sum(Xtrain ** 2, axis=1, keepdims=True)
            K = sq_norm + sq_norm.T - 2 * Xtrain @ Xtrain.T
        else:
            Xtest = np.asarray(Xtest)
            if Xtrain.shape[1] != Xtest.shape[1]:
                raise ValueError("Feature dimensions mismatch.")
            sq_norm_train = np.sum(Xtrain ** 2, axis=1, keepdims=True).T
            sq_norm_test = np.sum(Xtest ** 2, axis=1, keepdims=True)
            K = sq_norm_test + sq_norm_train - 2 * Xtest @ Xtrain.T
        return np.exp(-K / (2 * sigma2))

    @staticmethod
    def _weight_function(residual, weight_type):
        """Compute weights for iterative reweighting."""
        e = np.asarray(residual).ravel()
        n = len(e)
        Beta = np.zeros(n)
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

    # ------------------ Public methods ------------------

    def set_basis(self, Xsub):
        """Set pre‑selected basis vectors (must be scaled)."""
        self.Xsub = np.asarray(Xsub)
        self.num_basis = self.Xsub.shape[0]

    def fit(self, X, y):
        """
        Train the model.

        Parameters
        ----------
        X : 2D array, shape (n_samples, n_features)
            Expanded feature matrix (e.g., from expand_features).
        y : 1D array, target values.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same length.")

        # Preprocess (standardize)
        if self.preprocess == 'on':
            self.mean_Xtr = np.mean(X, axis=0)
            self.std_Xtr = np.std(X, axis=0, ddof=0)
            self.std_Xtr[self.std_Xtr == 0] = 1.0
            self.mean_ytr = np.mean(y)
            self.std_ytr = np.std(y, ddof=0)
            if self.std_ytr == 0:
                self.std_ytr = 1.0

            X_scaled = (X - self.mean_Xtr) / self.std_Xtr
            y_scaled = (y - self.mean_ytr) / self.std_ytr
        else:
            self.mean_Xtr = None
            self.std_Xtr = None
            self.mean_ytr = None
            self.std_ytr = None
            X_scaled = X
            y_scaled = y

        self.scaled_Xtr = X_scaled
        self.scaled_ytr = y_scaled

        # Select basis if not already set
        if self.Xsub is None:
            rng = check_random_state(self.random_state)
            idx = rng.permutation(X_scaled.shape[0])[:self.num_basis]
            self.Xsub = X_scaled[idx, :]
        else:
            # Ensure Xsub is compatible (should be already scaled)
            if self.preprocess == 'on':
                # Xsub should have been stored in scaled space
                pass

        # Robust training
        self._train_robust()

    def _train_robust(self):
        """Iteratively reweighted least squares training."""
        Xs = self.Xsub
        sig2 = self.sigma2
        Xtrain = self.scaled_Xtr
        ytrain = self.scaled_ytr
        gam = self.gamma

        # Kernel matrices
        K = self._kernelmatrix(Xs, sig2)
        Ke = self._kernelmatrix(Xs, sig2, Xtrain).T   # shape (nb, n)

        nb = K.shape[0]
        KA = np.zeros((nb + 1, nb + 1))
        KA[:nb, :nb] = K
        KI = np.vstack((Ke, np.ones((1, Ke.shape[1]))))

        B = np.eye(Xtrain.shape[0])
        # A_const = (KA + KA.T) / (2.0 * gam)
        A_const = (KA + KA.T)*gam / 2.0

        residual_old = np.zeros(len(ytrain))
        cond_best = np.inf
        count = 0
        max_iter = 500

        for it in range(max_iter):
            A = A_const + KI @ B @ KI.T
            lambda_reg = 1e-10   # 尝试的值，可根据结果调整
            A_reg = A + lambda_reg * np.eye(A.shape[0])
            b = KI @ B @ ytrain
            model_pars = solve(A_reg, b, assume_a='sym')
            self.model_pars = model_pars

            # Predict on training data (already scaled)
            ypred = self.predict(self.scaled_Xtr, expand=False, scale=False)
            residual_new = ypred - ytrain

            cond1 = np.linalg.norm(residual_new - residual_old)
            cond2 = np.mean(residual_new ** 2)

            if cond1 < self.tol or cond2 < self.tol:
                print(f"Convergence reached after {it+1} iteration(s)")
                break

            if cond1 <= cond_best:
                cond_best = cond1
            else:
                count += 1
            if count > self.patience_count:
                print(f"Early stopping after {it+1} iteration(s)")
                break

            residual_old = residual_new
            Beta = self._weight_function(residual_old, self.weight_type)
            B = np.diag(Beta)

            if it == max_iter - 1:
                print(f"Warning: Max iterations reached, f(x) = {cond1:.6f}, residual norm = {cond2:.6f}")

    def predict(self, X, expand=True, scale=True):
        """
        Predict outputs.

        Parameters
        ----------
        X : 1D or 2D array
            If expand=True, X is assumed to be raw signal (1D).
            If expand=False, X is already expanded feature matrix.
        expand : bool, whether to apply hysteresis expansion (needs thresholds set)
        scale : bool, whether to apply standardization (should be True for new data)

        Returns
        -------
        ypred : 1D array
        """
        if expand:
            if self.thresholds is None:
                raise ValueError("Thresholds not set. Provide thresholds or set expand=False.")
            X_exp = self.expand_features(X, self.thresholds, self.operator_type)
        else:
            X_exp = np.asarray(X)

        if scale and self.preprocess == 'on':
            X_scaled = (X_exp - self.mean_Xtr) / self.std_Xtr
        else:
            X_scaled = X_exp

        Kt = self._kernelmatrix(self.Xsub, self.sigma2, X_scaled)
        beta = self.model_pars[:-1]
        bias = self.model_pars[-1]
        ypred_scaled = Kt @ beta + bias

        if scale and self.preprocess == 'on':
            ypred = ypred_scaled * self.std_ytr + self.mean_ytr
        else:
            ypred = ypred_scaled
        return ypred

    def score(self, X, y, metric='R2', expand=True, scale=True):
        """Compute R² or NRMSE."""
        ypred = self.predict(X, expand=expand, scale=scale)
        y = np.asarray(y).ravel()
        if metric.upper() == 'R2':
            sse = np.sum((y - ypred) ** 2)
            sst = np.sum((y - np.mean(y)) ** 2)
            return 1 - sse / sst
        elif metric.upper() == 'NRMSE':
            rmse = np.sqrt(np.mean((y - ypred) ** 2))
            return rmse / (np.max(y) - np.min(y))
        else:
            raise ValueError("Metric must be 'R2' or 'NRMSE'")