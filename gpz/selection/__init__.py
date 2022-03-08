from . import lik, loo
       
selection_methods_dict = {
    "mle"           : lik.NLL,
    "loo-mse"       : loo.MSE,
    "loo-crps"      : loo.CRPS,
    "loo-nlpd"      : loo.NLPD
}