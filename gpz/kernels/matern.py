import numpy as np
from scipy import special
      
from .stationary import Stationary
        
class Matern32(Stationary):
    """
    The value of the kernel for a normalised distance :math:`r` is computed as
    
    .. math::
        
        k_\theta(r) = \sigma^2 (1 + \sqrt{3}r) e^{-\sqrt{3}r}
    """
    def __init__(self, dim = 1, sigma = 1, theta = 1, **kwargs):
        Stationary.__init__(self, dim, sigma, theta, **kwargs)
        
    def _k(self, r):
        r = np.array(r)
        return (1 + np.sqrt(3)*r) * np.exp(-np.sqrt(3)*r)
        
    def _dk_dr(self, r):
        r = np.array(r)
        return -3 * r * np.exp(-np.sqrt(3)*r)
        
class Matern52(Stationary):
    """
    The value of the kernel for a normalised distance :math:`r` is computed as
    
    .. math::
        
        k_\theta(r) = \sigma^2 (1 + \sqrt{5}r + \frac{5r^2}{3}) e^{-\sqrt{5}r}
    """
    def __init__(self, dim = 1, sigma = 1, theta = 1, **kwargs):
        Stationary.__init__(self, dim, sigma, theta, **kwargs)
        
    def _k(self, r):
        r = np.array(r)
        return (1 + np.sqrt(5)*r + 5*r**2/3) * np.exp(-np.sqrt(5)*r)
        
    def _dk_dr(self, r):
        r = np.array(r)
        return (10/3*r - 5*r - 5*np.sqrt(5)/3*r**2) * np.exp(-np.sqrt(5)*r)
        
class Matern(Stationary):
    """
    The value of the kernel for a normalised distance :math:`r` is computed as
    
    .. math::
        
        k_\theta(r) = ...
    """
    def __init__(self, dim = 1, sigma = 1, theta = 1, order = 2, **kwargs):
        Stationary.__init__(self, dim, sigma, theta, **kwargs)
        self.__set_order(order)
        
    def __set_order(self, order):
        try:
            self.__order = int(order)
            self.__k = lambda r : special.kn(self.__order, r)
        except:
            self.__order = float(order)
            self.__k = lambda r : special.kv(self.__order, r)
        finally:
            self.__dk = lambda r : special.kvp(self.__order, r)
        
    def _k(self, r):
        r = np.array(r)
        return self.__k(r)
        
    def _dk_dr(self, r):
        r = np.array(r)
        return self.__dk(r)
        
    def to_dict(self):
        output_dict = Stationary.to_dict(self)
        output_dict["order"] = self.__order
        return output_dict