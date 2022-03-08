from . import constant, exponential, matern, gaussian
from .constant import Constant
from .exponential import Exponential
from .matern import Matern32, Matern52, Matern
from .gaussian import Gaussian
       
kernels_dict = {
    "constant"      : Constant,
    "exponential"   : Exponential,
    "matern32"      : Matern32,
    "matern52"      : Matern52,
    "matern"        : Matern,
    "gaussian"      : Gaussian
}