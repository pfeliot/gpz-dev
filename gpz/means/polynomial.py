import itertools

import numpy as np
from scipy.special import comb

from .additive import Additive

def combinations(dim, degree, interaction_only = False, include_bias = True):
    comb = itertools.combinations if interaction_only else itertools.combinations_with_replacement
    start = 0 if include_bias else 1
    return itertools.chain.from_iterable(comb(range(dim), i) for i in range(start, degree + 1))

def combinatorial_terms(dim, degree, interaction_only = False, include_bias = True):
    cc = combinations(dim, degree, interaction_only, include_bias)
    powers = np.vstack([np.bincount(c, minlength = dim) for c in cc])
    terms = []
    for power in powers:
        if np.any(power > 0):
            index = power > 0
            terms.append(lambda X, power = power, index = index : np.prod(np.power(X[:, index], power[index]), axis = 1))
        else:
            terms.append(lambda X : np.ones(X.shape[0]))
    return terms
        
class Polynomial(Additive):
    def __init__(self, beta, dim, degree):
        nterms = comb(degree + dim, dim)
        assert len(beta) == nterms, "Wrong number of coefficients ({} instead of {})".format(len(beta), nterms)
        self.dim = dim
        self.degree = degree
        Additive.__init__(self, beta, combinatorial_terms(self.dim, self.degree))
        
    @property
    def dim(self):
        return self.__dim
        
    @property
    def degree(self):
        return self.__degree
        
    @dim.setter
    def dim(self, d):
        d = int(d)
        if d > 0:
            self.__dim = d
        else:
            raise ValueError("Dimension should be strictly positive.")
            
    @degree.setter
    def degree(self, o):
        self.__degree = int(o)