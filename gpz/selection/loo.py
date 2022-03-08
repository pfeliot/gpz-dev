import numpy as np
from scipy import linalg
from scipy.stats import norm

from .abc import SelectionMethod
from .mixins import StationaryMixin, MeanFuncUpdateMixin
from ..kernels.stationary import Stationary

def mse(z, mu, sigma2):
    return (z-mu)**2
    
def nlpd(z, mu, sigma2):
    return 0.5 * np.log (2 * np.pi * sigma2) + 0.5 * (mu - z)**2/sigma2
    
def crps(z, mu, sigma2):
    sigma = np.sqrt(sigma2)
    alpha = (z - mu) / sigma
    return (z-mu) * (2 * norm.cdf(alpha) - 1) + 2 * sigma * norm.pdf(alpha) - sigma / np.sqrt(np.pi)

class LOO(SelectionMethod, StationaryMixin, MeanFuncUpdateMixin):
    def __init__(self, model, X, y, **options):
        SelectionMethod.__init__(self, model, X, y, **options)
        if isinstance(self.model.covfunc, Stationary):
            StationaryMixin.__init__(self)
            self.jac = None
        else:
            raise NotImplementedError()
            
    @property 
    def scoring_rule(self):
        return self.__class__._scoring_rule
    
    def default_options(self):
        return {}
        
    def objfunc_stationary(self, x):
        self.model.covfunc.sigma, self.model.covfunc.theta, *_ = self._params_transform_stationary(x)
        mu, sigma2 = self.compute_loo()
        return 1 / self.y.shape[0] * np.sum(self.scoring_rule(self.y, mu, sigma2))
        
    def jac_stationary(self, x):
        raise NotImplementedError()
        
    def compute_loo(self):
        n = self.X.shape[0]
        K, L, Kinv = self.cholesky_computations(compute_Kinv = True)
        b = self.meanfunc_update(L, compute_b = True)
        mu = self.y - linalg.cho_solve((L, False), b) / np.diag(Kinv)
        sigma2 = 1 / np.diag(Kinv)
        return mu, sigma2

class MSE(LOO):
    _scoring_rule = mse

class NLPD(LOO):
    _scoring_rule = nlpd

class CRPS(LOO):
    _scoring_rule = crps