import numpy as np

from ..abc import CovarianceFunction

class Stationary(CovarianceFunction):
    def __init__(self, dim = 1, sigma = 1, theta = 1, **kwargs):
        CovarianceFunction.__init__(self)
        self.dim   = dim
        self.sigma = sigma
        self.theta = theta
        
    @property
    def dim(self):
        return self.__dim
        
    @property
    def sigma(self):
        return self.__sigma
        
    @property
    def variance(self):
        return self.__sigma ** 2
        
    @property
    def theta(self):
        return self.__theta
        
    @property
    def anisotropic(self):
        return hasattr(self.theta, "__iter__")
        
    @dim.setter
    def dim(self, d):
        self.__dim = int(d)
        
    @sigma.setter
    def sigma(self, sigma):
        sigma = float(sigma)
        if sigma > 0:
            self.__sigma = sigma
        else:
            raise ValueError("Sigma should be a strictly positive number.")
            
    @theta.setter
    def theta(self, theta):
        if hasattr(theta, "__iter__"):
            self.__set_anisotropic_theta(theta)
        else:
            self.__set_isotropic_theta(theta)
            
    #------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, input_dict):
        class_name = input_dict["class"]
        if cls.__name__ == class_name:
            return cls(**input_dict)
        else:
            return CovarianceFunction.from_dict(input_dict)
        
    def to_dict(self):
        output_dict = CovarianceFunction.to_dict(self)
        output_dict["dim"] = self.dim
        output_dict["sigma"] = self.sigma
        output_dict["theta"] = self.theta.tolist() if self.anisotropic else self.theta
        return output_dict 
        
    #------------------------------------------------------------------------------
    def _call__xx(self, x):
        X = np.array(x, ndmin = 2)
        r = self.__scaled_norm_XX(X)
        return self.variance * self._k(r)
        
    def _call__xy(self, x, y):
        X = np.array(x, ndmin = 2)
        Y = np.array(y, ndmin = 2)
        r = self.__scaled_norm_XY(X, Y)
        return self.variance * self._k(r)
        
    def compute_dK_dtheta(self, x):
        X = np.array(x, ndmin = 2)
        r = self.__scaled_norm_XX(X)
        dK_dtheta = self.variance * self.__dr_dtheta(X) * self._dk_dr(r)
        return dK_dtheta
            
    def _k(self, r):
        raise NotImplementedError
            
    def _dk_dr(self, r):
        raise NotImplementedError
        
    def __dr_dtheta(self, x):
        if self.anisotropic:
            r = self.__scaled_norm_XX(x)
            return -np.array([np.square(x[:,q:q+1] - x[:,q:q+1].T)/self.theta[q]**3 for q in range(self.dim)])/np.where(r != 0., r, np.inf)
        else:
            return -self.__scaled_norm_XX(x)/self.theta
            
    #------------------------------------------------------------------------------
    def __set_anisotropic_theta(self, theta):
        theta = np.array(theta).flatten()
        if not len(theta) == self.dim:
            raise ValueError("dim mismatch between theta ({}) and dim ({})".format(len(theta), self.dim))
        if np.all(theta > 0):
            self.__theta = theta
        else:
            raise ValueError("The theta should be positive numbers.")
            
    def __set_isotropic_theta(self, theta):
        theta = float(theta)
        if theta > 0:
            self.__theta = theta
        else:
            raise ValueError("The theta should be positive numbers.")
            
    #------------------------------------------------------------------------------
    def __scaled_norm_XX(self, X):
        if self.anisotropic:
            return self.__unscaled_norm_XX(X/self.theta)
        else:
            return self.__unscaled_norm_XX(X)/self.theta
            
    def __scaled_norm_XY(self, X, Y):
        if self.anisotropic:
            return self.__unscaled_norm_XY(X/self.theta, Y/self.theta)
        else:
            return self.__unscaled_norm_XY(X, Y)/self.theta
            
    def __unscaled_norm_XX(self, X):
        Xsq = np.sum(np.square(X),1)
        rsq = -2.*np.dot(X, X.T) + (Xsq[:,None] + Xsq[None,:])
        rsq = np.clip(rsq, 0, np.inf)
        return np.sqrt(rsq)
            
    def __unscaled_norm_XY(self, X, Y):
        Xsq = np.sum(np.square(X),1)
        Ysq = np.sum(np.square(Y),1)
        rsq = -2.*np.dot(X, Y.T) + (Xsq[:,None] + Ysq[None,:])
        rsq = np.clip(rsq, 0, np.inf)
        return np.sqrt(rsq)
        