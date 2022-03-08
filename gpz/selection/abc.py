from abc import ABCMeta, abstractmethod

from scipy.optimize import minimize

from .. import util
from ..abc import CovarianceFunction, MeanFunction

class SelectionMethod():
    __metaclass__ = ABCMeta
    
    def __init__(self, model, X, y, **options):
        self.set_options(options)
        self.set_X_y(X, y)
        self.model = model
        self.jac = None
        self.hess = None
        
    @property
    def X(self):
        return self.__X
        
    @property
    def y(self):
        return self.__y
        
    @property
    def model(self):
        return self.__model
        
    @model.setter
    def model(self, model):
        if not hasattr(model, "covfunc") and isinstance(model.covfunc, CovarianceFunction):
            raise TypeError("Invalid covariance function")
        if not hasattr(model, "meanfunc") and (isinstance(model.meanfunc, MeanFunction) or model.meanfunc is None):
            raise TypeError("Invalid mean function")
        self.__model = model
        
    def set_X_y(self, X, y):
        X, y = util.check_X_y(X, y)
        self.__X = X
        self.__y = y
        
    def set_options(self, options):
        self.options = self.default_options()
        self.options.update(options)
        
    @classmethod
    def run(cls, model, X, y, **options):
        options = dict(options)
        optimizer_options = options.pop("optimizer") if "optimizer" in options else {}
        method = cls(model, X, y, **options)
        result = minimize(method.objfunc, method.x0(), jac = method.jac, hess = method.hess, **optimizer_options)
        return method.on_optimization_end(result)
    
    @abstractmethod
    def objfunc(self, x):
        raise NotImplementedError()
        
    @abstractmethod
    def x0(self):
        raise NotImplementedError()
    
    def default_options(self):
        return {}
        
    def on_optimization_end(self, optimization_result):
        return self.model.posterior(self.X, self.y)