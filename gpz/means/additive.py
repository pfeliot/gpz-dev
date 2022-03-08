import binascii

import numpy as np

from ..abc import MeanFunction
    
class Additive(MeanFunction):
    def __init__(self, beta, terms):
        self.beta = beta
        self.terms = terms
        
    @property
    def beta(self):
        return self.__beta
        
    @property
    def terms(self):
        return self.__terms
        
    @beta.setter
    def beta(self, beta):
        beta = np.array(beta)
        self.__beta = beta.flatten()
        
    @terms.setter
    def terms(self, terms):
        terms = list(terms)
        if all([callable(term) for term in terms]):
            self.__terms = terms
        else:
            raise ValueError("Argument should be a sequence of callable terms.")
        
    def __call__(self, x):
        X = np.array(x, ndmin = 2)
        Y = np.array([term(X).flatten() for term in self.terms])
        return np.dot(self.beta, Y)
        
    @classmethod
    def from_dict(cls, input_dict):
        beta = input_dict["beta"]
        terms = [binascii.a2b_base64(term) for term in input_dict["terms"]]
        return cls(beta, terms)
        
    def to_dict(self):
        output_dict = MeanFunction.to_dict(self)
        output_dict["beta"] = self.beta.tolist()
        output_dict["terms"] = [binascii.b2a_base64(term) for term in self.terms]
        return output_dict
    
        
    