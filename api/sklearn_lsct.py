import tqdm
import numpy as np
from scipy.optimize import minimize
from sklearn.base import TransformerMixin, BaseEstimator, _fit_context
from sklearn.utils.validation import check_is_fitted
from helper.utils import Timer

class SklearnLSCTSplitter(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "maturities": [list],
        "n_factors": [int],
        "lambdas": [float, int],
        "verbose": [int],
    }

    def __init__(self, maturities: list[int|float]=None, n_factors:int = 4, lambdas: float|int = 0.5, verbose: int = 0):
        if n_factors>4 or n_factors<1:
            raise ValueError("n_factors has to be between 1 and 4.")
        self.n_factors = n_factors
        self.lambdas = lambdas
        self.maturities = maturities
        self.verbose = verbose
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X = None, y = None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        load_L = np.ones(len(self.maturities)).tolist()
        load_S = [(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) for t in self.maturities]
        load_C = [(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) - np.exp(-self.lambdas*t) for t in self.maturities]
        load_T = [2*(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) - np.exp(-self.lambdas*t)*(self.lambdas*t + 2) for t in self.maturities]
        self.loadings_ = np.array([load_L, load_S, load_C, load_T])
        self.loadings_ = self.loadings_[:self.n_factors]
        # Return the transformer
        return self

    def __objfunc(self, factors, true_ys):
        """
        Objective function used in optimization.

        Args:
        - factors (array): Model factors.
        - trueys (array): True values.

        Returns:
        - float: Sum of squared differences between true and predicted values.
        """
        factors = np.expand_dims(factors, axis=0)
        return np.nansum((true_ys - factors.dot(self.loadings_))**2)
    
    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        if self.verbose>0:
            print('Run LSCT splitter optimization.')
        with Timer('LSCT Optimization', display = self.verbose>0):
            values = X
            factors = []
            for row in tqdm.tqdm(values, disable = self.verbose < 2):
                res = minimize(lambda x: self.__objfunc(x, row), x0 = np.zeros(self.n_factors))
                factors.append(res.x)
            factors = np.array(factors)
        return factors
    
    def inverse_transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        return X.dot(self.loadings_)
    
    def get_feature_names_out(self, input_features=None):
        feature_names = ['level', 'slope', 'curvature', 'twist']
        return feature_names[:self.n_factors]
    
