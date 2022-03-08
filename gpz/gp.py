import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import linalg
from scipy.optimize import minimize_scalar

from .abc import CovarianceFunction, MeanFunction
from .kernels import kernels_dict
from .means import means_dict
from .selection import selection_methods_dict
from . import util

class GP():
    def __init__(self, covfunc, meanfunc = None, noisevar = 0, autojitter = 5, **kwargs):
        """Creates an instance of Gaussian process (GP) model with covariance function `covfunc` and 
        optional mean function `meanfunc`. Observational noise can be specified through its variance 
        `noisevar`.

        Parameters
        ----------
        covfunc : CovarianceFunction
            The model's covariance function.
            
        meanfunc : MeanFunction or None, optional (default = None)
            The model's mean function.
            
        noisevar : float, optional (default = 0)
            The model's observational noise variance.
            
        autojitter : int, optional (default = 5)
            Number of trials for the `util.jitchol` algorithm.
        """
        self.covfunc = covfunc
        self.meanfunc = meanfunc
        self.noisevar = noisevar
        self.__autojitter = int(autojitter)
        
    @property
    def covfunc(self):
        return self.__covfunc
        
    @property
    def meanfunc(self):
        return self.__meanfunc
        
    @property
    def noisevar(self):
        return self.__noisevar
        
    @covfunc.setter
    def covfunc(self, k):
        if k is None:
            self.__covfunc = self.__default_kernel()
        elif isinstance(k, str):
            self.__covfunc = self.__str_kernel(k)
        elif isinstance(k, CovarianceFunction):
            self.__covfunc = k
        else:
            raise TypeError("Invalid argument.")
        
    @meanfunc.setter
    def meanfunc(self, m):
        if m is None:
            self.__meanfunc = None
        elif isinstance(m, str):
            self.__meanfunc = self.__str_mean(m)
        elif isinstance(m, MeanFunction):
            self.__meanfunc = m
        else:
            raise TypeError("Invalid argument.")
        
    @noisevar.setter
    def noisevar(self, s):
        s = float(s)
        if s >= 0:
            self.__noisevar = s
        else:
            raise TypeError("Invalid argument.")
        
    #------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, input_dict):
        """Creates an instance of Gaussian process (GP) from a configuration dictionary,
        usually obtained with a call to the `GP.to_dict` method.

        Parameters
        ----------
        input_dict : dict
            A configuration dictionary.
            
        Returns
        -------
        gp : GP
            A GP object with mean and covariance functions built from `input_dict`.
        """
        covfunc = CovarianceFunction.from_dict(input_dict["covfunc"])
        meanfunc = MeanFunction.from_dict(input_dict["meanfunc"]) if input_dict["meanfunc"] is not None else None
        noisevar = input_dict["noisevar"] if "noisevar" in input_dict else {}
        autojitter = input_dict["autojitter"] if "autojitter" in input_dict else {}
        
        return cls(covfunc, meanfunc, noisevar, autojitter)
        
    def to_dict(self):
        """Serialization of this `GP` object to a dictionary that can be saved using, e.g., the `json` package.
            
        Returns
        -------
        output_dict : dict
            A dictionary representation of this object.
        """
        output_dict = {
            "covfunc" : self.covfunc.to_dict()
            }
        output_dict["meanfunc"] = None if self.meanfunc is None else self.meanfunc.to_dict()
        output_dict["noisevar"] = self.noisevar
        output_dict["autojitter"] = self.__autojitter
        return output_dict
        
    def copy(self):
        """Object copy.
            
        Returns
        -------
        gp : GP
            A copy of this object.
        """
        m = None if self.meanfunc is None else self.meanfunc.copy()
        return GP(self.covfunc.copy(), m, self.noisevar, self.__autojitter)
    
    def posterior(self, X, y):
        """Posterior process using data `X` and `y`.

        Parameters
        ----------
        X : numpy.ndarray
            Input values with shape (n, d).
            
        y : numpy.ndarray
            Output values with shape (n,).
            
        Returns
        -------
        posterior : GP
            A GP object with posterior mean and covariance functions.
        """
        X, y = util.check_X_y(X, y)
        K = self.covfunc(X)
        C = util.jitchol(K + self.noisevar * np.eye(K.shape[0]), self.__autojitter)
        b = y - self.meanfunc(X) if self.meanfunc is not None else y
        posterior_kernel = _PosteriorCov(self.covfunc, C, X)
        posterior_mean   = _PosteriorMean(self.meanfunc, self.covfunc, C, X, b)
        return GP(posterior_kernel, posterior_mean)
        
    def sample(self, X, N):
        """For a vector X = (X_1, ..., X_n), this method generates `N` realisations of the 
        random vector (gp(X_1), ..., gp(X_n)).

        Parameters
        ----------
        X : numpy.ndarray with shape (n, d)
            Array with shape (n, d). The rows are the sites at which realisations have to be drawn.
            
        N : int
            Number of realisations to draw.
            
        Returns
        -------
        S : numpy.ndarray
            An array with shape (N, n).
        """
        X = util.check_X(X)
        N = int(N)
        m = self.meanfunc(X) if self.meanfunc is not None else np.zeros(X.shape[0])
        return np.random.multivariate_normal(m, self.covfunc(X), N)
        
    def optimize(self, X, y, method = "mle", **method_options):
        """This method is used to optimize the model parameters, using data `X` and `y`
        and the method `method`.

        Parameters
        ----------
        X : numpy.ndarray
            Input values with shape (n, d).
            
        y : numpy.ndarray
            Output values with shape (n,).
            
        method : str, optional (default = "mle")
            The method to be used. Acceptable values are listed 
            in `gpz.selection.selection_methods_dict`.
            
        Returns
        -------
        posterior : GP
            A `GP` object with posterior mean and covariance functions with parameters 
            optimized for the data `X` and `y`.
        """
        selection_method = selection_methods_dict[method]
        return selection_method.run(self, X, y, **method_options)
    
    #------------------------------------------------------------------------------
    def __default_kernel(self):
        return kernels_dict["matern52"]()
        
    def __str_kernel(self, k):
        k = str(k)
        return kernels_dict[k.lower()]()
        
    def __str_mean(self, m):
        m = str(m)
        return means_dict[m.lower()]()
        
class _PosteriorCov(CovarianceFunction):
    def __init__(self, covfunc, C, X):
        CovarianceFunction.__init__(self)
        self.__covfunc = covfunc
        self.__C = C
        self.__X = X
        
    def __call__(self, x, y = None):
        if y is not None:
            return self.__covfunc(x, y) - np.dot(self.__covfunc(self.__X, x).T, linalg.cho_solve((self.__C, False), self.__covfunc(self.__X, y)))
        else:
            kx = self.__covfunc(self.__X, x)
            return self.__covfunc(x, y) - np.dot(kx.T, linalg.cho_solve((self.__C, False), kx))
            
    @classmethod
    def from_dict(cls, input_dict):
        class_name = input_dict["class"]
        if cls.__name__ == class_name:
            covfunc = CovarianceFunction.from_dict(input_dict["covfunc"])
            C = np.array(output_dict["C"], ndmin = 2)
            X = np.array(output_dict["X"], ndmin = 2)
            return cls(covfunc, C, X)
        else:
            return CovarianceFunction.from_dict(input_dict)
    
    def to_dict(self):
        output_dict = CovarianceFunction.to_dict(self)
        output_dict["covfunc"] = self.__covfunc.to_dict()
        output_dict["C"] = self.__C.tolist()
        output_dict["X"] = self.__X.tolist()
        return output_dict
        
class _PosteriorMean(MeanFunction):
    def __init__(self, meanfunc, covfunc, C, X, y):
        MeanFunction.__init__(self)
        self.__meanfunc = meanfunc
        self.__covfunc = covfunc
        self.__C = C
        self.__X = X
        self.__y = y
        self.__alpha = linalg.cho_solve((self.__C, False), self.__y)
        
    def __call__(self, x):
        m = np.dot(self.__alpha, self.__covfunc(self.__X, x))
        return self.__meanfunc(x) + m.flatten() if self.__meanfunc is not None else m.flatten()
            
    @classmethod
    def from_dict(cls, input_dict):
        class_name = input_dict["class"]
        if cls.__name__ == class_name:
            meanfunc = MeanFunction.from_dict(input_dict["meanfunc"])
            covfunc = CovarianceFunction.from_dict(input_dict["covfunc"])
            C = np.array(output_dict["C"], ndmin = 2)
            X = np.array(output_dict["X"], ndmin = 2)
            y = np.array(output_dict["y"])
            return cls(meanfunc, covfunc, C, X, y)
        else:
            return MeanFunction.from_dict(input_dict)
    
    def to_dict(self):
        output_dict = CovarianceFunction.to_dict(self)
        output_dict["meanfunc"] = self.__meanfunc.to_dict()
        output_dict["covfunc"] = self.__covfunc.to_dict()
        output_dict["C"] = self.__C.tolist()
        output_dict["X"] = self.__X.tolist()
        output_dict["y"] = self.__y.tolist()
        return output_dict
        