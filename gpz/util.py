import numpy as np
from scipy import linalg

def jitchol(K, numtry = 5):
    try:
        C, _ = linalg.cho_factor(K)
        return C
    except:
        pass
    jitter = np.clip(np.diag(K), 0, np.inf).mean() * np.logspace(1e-14, 1e-3, numtry)
    i = 0
    while i < numtry:
        try:
            C, _ = linalg.cho_factor(K + jitter[i] * np.eye(K.shape[0]))
            return C
        except:
            pass
        i += 1
    raise linalg.LinAlgError("not positive definite, even with jitter.")
        
def check_X_y(X, y):
    X = check_X(X)
    y = np.array(y).ravel()
    if X.shape[0] != len(y):
        raise ValueError("Missmatching shapes between X {} and y {}.".format(X.shape, y.shape))
    return X, y
    
def check_X(X):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape((-1,1))
    return X