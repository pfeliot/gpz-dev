from . import additive, constant, polynomial
from .constant import Constant
from .polynomial import Polynomial
from .additive import Additive
       
means_dict = {
    "constant" : Constant,
    "polynomial" : Polynomial,
    "additive" : Additive
}