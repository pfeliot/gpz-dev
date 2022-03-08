import numpy as np
from scipy import linalg

from ..means import Constant, Additive
from .. import util

class ComputationMixin():
    def cholesky_computations(self, compute_Kinv = False):
        K = self.model.covfunc(self.X) + self.model.noisevar * np.eye(self.X.shape[0])
        L = util.jitchol(K)
        Kinv = None
        if compute_Kinv:
            Kinv, _ = linalg.lapack.dpotri(L, lower = 1)
        return K, L, Kinv

class StationaryMixin(ComputationMixin):
    def __init__(self):
        self.objfunc = self.objfunc_stationary
        self.x0      = self.x0_stationary
        self.jac     = self.jac_stationary
        self.on_optimization_end = self.on_optimization_end_stationary
        
    def x0_stationary(self):
        sigma2 = np.var(self.y) * 0.5
        if self.model.covfunc.anisotropic:
            theta0 = (np.amax(self.X, axis = 0) - np.amin(self.X, axis = 0)) / self.X.shape[0]
        else:
            theta0 = np.array([np.mean(np.amax(self.X, axis = 0) - np.amin(self.X, axis = 0)) / self.X.shape[0]])
        return [np.log(sigma2)] + np.log(theta0).tolist()
        
    def on_optimization_end_stationary(self, optimization_result):
        self.model.covfunc.sigma, self.model.covfunc.theta, *_ = self._params_transform_stationary(optimization_result.x)
        K, L, *_ = self.cholesky_computations()
        self.meanfunc_update(L)
        return self.model.posterior(self.X, self.y)
        
    def _params_transform_stationary(self, x):
        sigma = np.sqrt(np.exp(x[0]))
        theta = np.exp(x[1:])
        dsigma2_dx = np.exp(-x[0])
        dtheta_dx = np.exp(-np.array(x[1:]))
        return sigma, theta, dsigma2_dx, dtheta_dx
        
class MeanFuncUpdateMixin():
    def meanfunc_update(self, L, compute_b = False):
        if self.model.meanfunc is not None: 
            if isinstance(self.model.meanfunc, Constant):
                self._constant_meanfunc_update(L)
            elif isinstance(self.model.meanfunc, Additive):
                self._additive_meanfunc_update(L)
            else:
                raise NotImplementedError()
        b = None
        if compute_b:
            b = self.y - self.model.meanfunc(self.X) if self.model.meanfunc is not None else self.y
        return b
        
    def _constant_meanfunc_update(self, L):
        self.model.meanfunc.value = np.sum(linalg.cho_solve((L, False), self.y)) / np.sum(linalg.cho_solve((L, False), np.ones_like(self.y)))
        
    def _additive_meanfunc_update(self, L):
        F = np.array([term(self.X).flatten() for term in self.model.meanfunc.terms])
        KF = np.dot(F, linalg.cho_solve((L, False), F.T))
        LF = util.jitchol(KF)
        self.model.meanfunc.beta = linalg.cho_solve((LF, False), np.dot(F, linalg.cho_solve((L, False), self.y)))
    