import numpy as np

from ..abc import MeanFunction
    
class Constant(MeanFunction):
    def __init__(self, value):
        self.value = value
        
    @property
    def value(self):
        return self.__value
        
    @value.setter
    def value(self, value):
        self.__value = float(value)
        
    def __call__(self, x):
        X = np.array(x, ndmin = 2)
        return np.repeat(self.value, X.shape[0])
        
    @classmethod
    def from_dict(cls, input_dict):
        value = input_dict["value"]
        return cls(value)
        
    def to_dict(self):
        output_dict = MeanFunction.to_dict(self)
        output_dict["value"] = self.value
        return output_dict