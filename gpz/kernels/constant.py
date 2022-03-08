import numpy as np

from .stationary import Stationary
        
class Constant(Stationary):
    """
    The value of the kernel for a normalised distance :math:`r` is computed as
    
    .. math::
        
        k_\theta(r) = \sigma^2e^{-r}
    """
    def __init__(self, dim = 1, sigma = 1, theta = 1):
        Stationary.__init__(self, dim, sigma, theta)
        
    def _k(self, r):
        r = np.array(r)
        return np.ones(r.shape)