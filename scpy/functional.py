import torch
import numpy as np
import pandas as pd

from statsmodels.base.optimizer import Optimizer

from tqdm import tqdm

from typing import Any, Callable, Iterable, Optional

from .functions import *


# utils

def _summary(xname, params, bse, statistic, pvalues):
  
    if xname != None:
        assert len(xname) == len(params), 'The number of features in x is different from xname: {} != {}'.format(len(xname), len(params))
        
    summary = pd.DataFrame(
        [params, bse, statistic, pvalues],
        index=['Coef.', 'Std.Err.', 'z', 'P>|z|'],
        columns=xname
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
  
def _tensors(device, x=None, y=None):
  
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        elif isinstance(y, torch.Tensor):
            device = y.device
        else:
            raise(Exception('Need to specify the device when the input is not a tensor'))
    
    results = [device]
    
    if x is not None:
        if isinstance(x, torch.Tensor):
            results.append(x.to(device))
        else:
            results.append(torch.tensor(x, device=device))
        
    if y is not None:
        if isinstance(y, torch.Tensor):
            if not y.is_sparse:
                y = y.view((-1,))
            results.append(y.to(device))
        else:
            results.append(torch.tensor(y, device=device))
            
    return results
  
def _family(family: str):
    
    if family == 'negative_binomial':
        return negative_binomial
    elif family == 'poisson':
        return poisson
    else:
        raise(ValueError(family))
  
def _parallel(X, Y, family, device, xname, yname, disp, kwargs):
  
    if type(Y) == dict:
      client, offset = Y['data'], Y['offset']
      def loader(i):
          client.send(i + offset)
          return client.recv()
      Y = dataloader(loader, length=Y['length'])
      
    return fit_batch(X, Y, family, device, xname, yname, disp, **kwargs)
  
  
# functions  

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
        
    return params, bse, tvalues, pvalues


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
        
    return params, bse, tvalues, pvalues
  
  
# data
  
class Dataset(torch.utils.data.Dataset):
  
    def __init__(self, data, length=None):
        self.data = data
        self.length = length
        
        assert hasattr(self.data, '__len__') or self.length is not None, 'data has no len() and length is None'
        
    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.data)
      
    def __getitem__(self, index):
        if callable(self.data):
            return self.data(index)
        return self.data[index]
  
  
def dataloader(data, length=None, num_workers=0):
    dataset = Dataset(data, length)
    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=1, shuffle=False, pin_memory=True)
  
  
def dataserver(con, getter: Callable):
    while True:
        index = con.recv()
        if index is None:
            break
        con.send(getter(index))
  

# wrappers
  
def fit(x, y, family: str, device=None, xname=None, **kwargs):
  
    f = _family(family)
        
    device, x, y = _tensors(device, x, y)
    
    assert x.dim() == 2, 'x should be 2-dimensional: {}'.format(x.shape)
    assert x.shape[0] == y.shape[0], 'x and y have different observations: {} != {}'.format(x.shape[0], y.shape[0])
        
    params, bse, statistic, pvalues = f(x, y, **kwargs)
    
    summary = _summary(xname, params, bse, statistic, pvalues)
        
    return summary
    
  
def fit_batch(X, Y, family: str, device=None, xnames=None, yname=None, disp=True, **kwargs):

    f = _family(family)
    
    if xnames is None:
        xnames = dict.fromkeys(range(len(X)), None)
    else:
        assert len(X) == len(xnames), 'The batch size of X is different from xnames: {} != {}'.format(len(X), len(xnames))
        
    if yname is None:
        yname = range(len(Y))
    else:
        assert len(Y) == len(yname), 'The batch size of Y is different from yname: {} != {}'.format(len(Y), len(yname))
    
    results = list()
    for target, y in zip(tqdm(yname, disable=(not disp)), Y):
      
        for (group, xname), x in zip(xnames, X):

            device, x, y = _tensors(device, x, y)

            params, bse, statistic, pvalues = f(x, y, **kwargs)

            summary = _summary(xname, params, bse, statistic, pvalues)
            summary['group'] = group
            summary['term'] = summary.index
            summary['target'] = target

            results.append(summary)
        
    results = pd.concat(results, axis=0)
    results.set_index(['group', 'term', 'target'], inplace=True)
    
    return results
