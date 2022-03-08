import sys
import importlib
from abc import ABCMeta, abstractmethod
    
class UnaryFunction():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError
    
class BinaryFunction():
    __metaclass__ = ABCMeta
    
    def __call__(self, x, y = None):
        if y is None:
            return self._call__xx(x)
        else:
            return self._call__xy(x, y)
        
    @abstractmethod
    def _call__xx(self, x):
        raise NotImplementedError
        
    @abstractmethod
    def _call__xy(self, x, y):
        raise NotImplementedError
        
class Serializable():
    __metaclass__ = ABCMeta
        
    @classmethod
    def from_dict(cls, input_dict):
        _cls_ = getattr(sys.modules[input_dict["module"]], input_dict["class"])
        return _cls_.from_dict(input_dict)
        
    def to_dict(self):
        return {"module" : self.__module__, "class" : self.__class__.__name__}
        
    def copy(self):
        return self.__class__.from_dict(self.to_dict())

class Composable():
    __metaclass__ = ABCMeta
    
    @property
    @abstractmethod
    def base_class(self):
        raise NotImplementedError
    
    def __add__(self, other):
        return self.__composite_objet(other, Add)
    
    def __sub__(self, other):
        return self.__composite_objet(other, Sub)
    
    def __mul__(self, other):
        return self.__composite_objet(other, Mul)
    
    def __truediv__(self, other):
        return self.__composite_objet(other, Div)
            
    __radd__ = __add__   
    __rsub__ = __sub__   
    __rmul__ = __mul__   
    __rtruediv__ = __truediv__  
            
    def __composite_objet(self, other, op):
        if isinstance(other, self.base_class):
            cls = Composable._create_composite_class(op, self.base_class)
            return cls(self, other)
        else:
            return NotImplemented
         
    @classmethod
    def _create_composite_class(cls, op, base_class):
        return type("Composite" + op.__name__ + base_class.__name__, (op, base_class), {"_type" : op.__name__})
        
class CovarianceFunction(BinaryFunction, Serializable, Composable):
    __metaclass__ = ABCMeta
    
    @property
    def base_class(self):
        return globals()["CovarianceFunction"]
        
class MeanFunction(UnaryFunction, Serializable, Composable):
    __metaclass__ = ABCMeta
    
    @property
    def base_class(self):
        return globals()["MeanFunction"]
        
class Composite(Composable):
    __metaclass__ = ABCMeta
         
    def __init__(self, l, r):
        if isinstance(l, self.base_class):
            self.__l = l
        else:
            raise TypeError("Left argument should be a {}.".format(self.base_class))
        if isinstance(r, self.base_class):
            self.__r = r
        else:
            raise TypeError("Right argument should be a {}.".format(self.base_class))
            
    @property
    def l(self):
        return self.__l
            
    @property
    def r(self):
        return self.__r
            
    @classmethod
    def from_dict(cls, input_dict):
        base_class = globals()[input_dict["base_class"]]
        op = globals()[input_dict["type"]]
        l = base_class.from_dict(input_dict["l"])
        r = base_class.from_dict(input_dict["r"])
        cls = Composable._create_composite_class(op, base_class)
        return cls(l, r)
        
    def to_dict(self):
        output_dict = Serializable.to_dict(self)
        output_dict["class"] = "Composite"
        output_dict["type"] = self._type
        output_dict["base_class"] = self.base_class.__name__
        output_dict["l"] = self.l.to_dict()
        output_dict["r"] = self.r.to_dict()
        return output_dict 
    
class Add(Composite):
    def __call__(self, *args, **kwargs):
        return self.l(*args, **kwargs) + self.r(*args, **kwargs)
        
class Sub(Composite):
    def __call__(self, *args, **kwargs):
        return self.l(*args, **kwargs) - self.r(*args, **kwargs)
        
class Mul(Composite):
    def __call__(self, *args, **kwargs):
        return self.l(*args, **kwargs) * self.r(*args, **kwargs)
        
class Div(Composite):
    def __call__(self, *args, **kwargs):
        return self.l(*args, **kwargs) / self.r(*args, **kwargs)