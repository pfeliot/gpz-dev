import numpy as np
      
from .stationary import Stationary
        
class Gaussian(Stationary):
    """
    The value of the kernel for a normalised distance :math:`r` is computed as
    
    .. math::
        
        k_\theta(r) = \sigma^2 e^{-\frac{r^2}{2}}
    """
    def __init__(self, dim = 1, sigma = 1, theta = 1):
        Stationary.__init__(self, dim, sigma, theta)
        
    def _k(self, r):
        r = np.array(r)
        return np.exp(-r**2/2)
        
    def _dk_dr(self, r):
        r = np.array(r)
        return -r * np.exp(-r**2/2)