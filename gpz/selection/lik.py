import numpy as np
from scipy import linalg

from .abc import SelectionMethod
from .mixins import StationaryMixin, MeanFuncUpdateMixin
from ..kernels.stationary import Stationary

class NLL(SelectionMethod, StationaryMixin, MeanFuncUpdateMixin):
    def __init__(self, model, X, y, **options):
        SelectionMethod.__init__(self, model, X, y, **options)
        if isinstance(self.model.covfunc, Stationary):
            StationaryMixin.__init__(self)
        else:
            raise NotImplementedError()
            
    def default_options(self):
        return {}
        
    def objfunc_stationary(self, x):
        self.model.covfunc.sigma, self.model.covfunc.theta, *_ = self._params_transform_stationary(x)
        K, L, *_ = self.cholesky_computations()
        b = self.meanfunc_update(L, compute_b = True)
        return 0.5 * (self.X.shape[0] * np.log(2*np.pi) + np.sum(np.log(np.diag(L))) + np.dot(b, linalg.cho_solve((L, False), b)))
        
    def jac_stationary(self, x):
        """See, e.g., http://www.gaussianprocess.org/gpml/chapters/RW5.pdf"""
        self.model.covfunc.sigma, self.model.covfunc.theta, dsigma2_dx, dtheta_dx = self._params_transform_stationary(x)
        K, L, Kinv = self.cholesky_computations(compute_Kinv = True)
        b = self.meanfunc_update(L, compute_b = True)
        dK_dtheta = self.model.covfunc.compute_dK_dtheta(self.X)
        alpha = linalg.cho_solve((L, False), b).reshape((-1,1))
        dL_dtheta = [-0.5 * np.trace(np.dot(np.dot(alpha, alpha.T) - Kinv, dK_dtheta[i])) for i in range(len(self.model.covfunc.theta))]
        dL_dsigma2 = 0.5/self.model.covfunc.variance * (K.shape[0] - self.model.noisevar * np.trace(Kinv) - np.dot(b, alpha) + self.model.noisevar * np.sum(alpha**2))
        return np.concatenate((dsigma2_dx * dL_dsigma2, dtheta_dx * dL_dtheta))
    