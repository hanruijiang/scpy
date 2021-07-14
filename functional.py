import torch
import numpy as np
import pandas as pd

from statsmodels.base.optimizer import Optimizer

from typing import Any, Callable, Iterable, Optional

from .functions import *


def _summary(params, bse, tvalues, pvalues):
  
    summary = pd.DataFrame(
        [params, bse, tvalues, pvalues],
        index=['Coef.', 'Std.Err.', 'z', 'P>|z|'],
        columns=['gRNA','guide_count','percent.mito','prep_batch','intercept']
    ).T
        
    return summary
  
def _optimize(f, score, hess, start_params, method: str, maxiter: int, **kwargs):
  
    if method == 'newton':
        def gradient(params, *args):
            return -score(params, *args)
        def hessian(params, *args):
            return -hess(params, *args)
    else:
        gradient = score
        hessian = hess
        
    with torch.no_grad():
      
        params = Optimizer()._fit(f, gradient, start_params, fargs=(), kwargs=kwargs,
             hessian=hessian, method=method, maxiter=maxiter, full_output=False,
             disp=False, callback=None, retall=False)[0]
    
    return params
    

def negative_binomial(x: torch.Tensor, y: torch.Tensor, method='bfgs', maxiter=35, full_output=True, **kwargs):

    start_params = [0] * (x.shape[1] - 1)
    if y.is_sparse:
        start_params.append(torch.log(torch.sum(y._values()) / y.shape[0]).cpu())
    else:
        start_params.append(torch.log(torch.mean(y)).cpu())  
    
    def optim_f(params, *args):
        return optim_nb_f(x, y, params)
    
    def optim_score(params, *args):
        return optim_nb_score(x, y, params)
    
    def optim_hessian(params, *args):
        return optim_nb_hessian(x, y, params)
        
    params = _optimize(optim_f, optim_score, optim_hessian, start_params, method, maxiter, **kwargs)
    
    if not full_output:
        return params
        
    with torch.no_grad():
        
        bse, tvalues, pvalues = stats_nb_t(x, y, params)
        
    return _summary(params, bse, tvalues, pvalues)


def poisson(x: torch.Tensor, y: torch.Tensor, method='newton', maxiter=35, full_output=True, **kwargs):
  
    start_params = [0] * (x.shape[1] - 1)
    if y.is_sparse:
        start_params.append(torch.log(torch.sum(y._values()) / y.shape[0]).cpu())
    else:
        start_params.append(torch.log(torch.mean(y)).cpu())
    
    def optim_f(params, *args):
        return optim_poi_f(x, y, params)
    
    def optim_score(params, *args):
        return optim_poi_score(x, y, params)
        
    def optim_hessian(params, *args):
        return optim_poi_hessian(x, y, params)
      
    params = _optimize(optim_f, optim_score, optim_hessian, start_params, method, maxiter, **kwargs)
        
    if not full_output:
        return params
      
    with torch.no_grad():
        
        bse, tvalues, pvalues = stats_poi_t(x, y, params)
        
    return _summary(params, bse, tvalues, pvalues)


  
