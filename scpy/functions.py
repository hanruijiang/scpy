import torch
import numpy as np
from scipy import stats

from typing import Any, Callable, Iterable, Optional
  
# optimization

## poisson

def optim_poi_f(x: torch.Tensor, y: torch.Tensor, params):
    xb = predict(x, params, linear=True)
    if y.is_sparse:
      nll = -poi_loglike_sparse(xb, y)
    else:
      nll = -torch.sum(poi_loglike_obs(xb, y))
    return nll.cpu().numpy()
  
def optim_poi_score(x: torch.Tensor, y: torch.Tensor, params):
    l = predict(x, params)
    if y.is_sparse:
      s = -poi_score_sparse(l, x, y)
    else:
      s = -poi_score(l, x, y)
    return s.cpu().numpy()
  
def optim_poi_hessian(x: torch.Tensor, y: torch.Tensor, params):
    l = predict(x, params)
    h = -poi_hessian(l, x)
    return h.cpu().numpy()

## negative binomial

def optim_nb_f(x: torch.Tensor, y: torch.Tensor, params):
    mu = predict(x, params)
    if y.is_sparse:
      nll = -nb_loglike_sparse(mu, y)# / x.shape[0]
    else:
      nll = -torch.sum(nb_loglike_obs(mu, y))
#       nll = -torch.mean(nb_loglike_obs(mu, y))
    return nll.cpu().numpy()
  
def optim_nb_score(x: torch.Tensor, y: torch.Tensor, params):
    mu = predict(x, params)
    if y.is_sparse:
      s = -nb_score_sparse(mu, x, y)# / x.shape[0]
    else:
      s = -torch.sum(nb_score_obs(mu, x, y), 0)
#       s = -torch.mean(nb_score_obs(mu, x, y), 0)
    return s.cpu().numpy()
  
def optim_nb_hessian(x: torch.Tensor, y: torch.Tensor, params):
    mu = predict(x, params)
    if y.is_sparse:
      h = -nb_hessian_sparse(mu, x, y)
    else:
      h = -nb_hessian(mu, x, y)
    return h.cpu().numpy()
  
# statistics

def stats_t(params, H: torch.Tensor, df: int):
    Hinv = hessian_inv(H)
    bse = torch.sqrt(torch.diag(Hinv)).cpu().numpy()
    tvalues = (params / bse)
    pvalues = stats.t.sf(np.abs(tvalues), df) * 2
    return bse, tvalues, pvalues
    

def stats_nb_t(x: torch.Tensor, y: torch.Tensor, params):
    mu = predict(x, params)
    if y.is_sparse:
      H = -1 * nb_hessian_sparse(mu, x, y)
    else:
      H = -1 * nb_hessian(mu, x, y)
    return stats_t(params, H, y.shape[0] - x.shape[1])

def stats_poi_t(x: torch.Tensor, y: torch.Tensor, params):
    l = predict(x, params)
    H = -1 * poi_hessian(l, x)
    return stats_t(params, H, y.shape[0] - x.shape[1])
    
    
# functions

def predict(x: torch.Tensor, params, linear=False, tensor=True):
    params = torch.tensor(params, device=x.device, dtype=x.dtype)[:x.shape[1]]
    pred = torch.matmul(x, params)
    if not linear:
      pred = torch.exp(pred)
    return pred if tensor else pred.cpu().numpy()

def hessian_inv(H: torch.Tensor):
    # TODO
    try:
      return torch.linalg.pinv(H.cpu()).to(H.device)
#       eigvals, eigvecs = torch.linalg.eigh(H.cpu())
#       Hinv = torch.matmul(torch.matmul(eigvecs, torch.diag(1.0 / eigvals)), eigvecs.T).to(H.device)
#       return (Hinv + Hinv.T) / 2.0
    except:
      return torch.ones_like(H) * float('Inf')
  
## poisson

def poi_loglike_obs(xb: torch.Tensor, y: torch.Tensor):
    llf = -torch.exp(xb) + y*xb - torch.special.gammaln(y + 1)
    return llf

def poi_score(l: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    return torch.matmul((y - l), x)

def poi_hessian(l: torch.Tensor, x: torch.Tensor):
    return -torch.matmul(l*x.T, x)

## negative binomial

def nb_loglike_obs(mu: torch.Tensor, y: torch.Tensor, alpha=1, Q=0):
    if alpha == 1 and Q == 0:
        prob = 1/(1+mu)
        llf = torch.log(prob) + y*torch.log(1-prob)
    else:
        size = 1/alpha * mu**Q
        prob = size/(size+mu)
        coeff = torch.special.gammaln(size+y) - torch.special.gammaln(y + 1) - torch.special.gammaln(size)
        llf = coeff + size*torch.log(prob) + y*torch.log(1-prob)
    return llf

def nb_score_obs(mu: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    mu = mu[:, None]
    y = y[:, None]
    dparams = x * (y-mu) / (mu+1)
    return dparams

def nb_hessian(mu: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    mu = mu[:,None]
    y = y[:,None]
    # for dl/dparams dparams
    dim = x.shape[1]
    hess_arr = torch.empty((dim, dim), device=x.device)
    const_arr = mu*(1+y)/(mu+1)**2
    for i in range(dim):
        for j in range(dim):
            if j > i:
                continue
            k = (-x[:,i,None] * x[:,j,None] * const_arr).sum(0)
            hess_arr[i,j] = k
            hess_arr[j,i] = k
    return hess_arr
    

# sparse functions

## poisson

def poi_loglike_sparse(xb: torch.Tensor, y: torch.Tensor):
    llf = -torch.sum(torch.exp(xb)) + torch.sum(y._values()*xb[y._indices()[0]]) - torch.sum(torch.special.gammaln(y._values() + 1))
    return llf
  
def poi_score_sparse(l: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    return torch.matmul(y, x) - torch.matmul(l, x)

## negative binomial

def nb_loglike_sparse(mu: torch.Tensor, y: torch.Tensor):
    prob = 1/(1+mu)
    llf = torch.sum(torch.log(prob)) + torch.sum(y._values()*torch.log(1-prob[y._indices()[0]]))
    return llf

def nb_score_sparse(mu: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    mu = mu[:, None]
    t = x / (mu+1)
    dparams = torch.sum(t[y._indices()[0]] * y._values()[:, None], 0) - torch.sum(t * mu, 0)
    return dparams

def nb_hessian_sparse(mu: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    mu = mu[:,None]
    dim = x.shape[1]
    hess_arr = torch.empty((dim, dim), device=x.device)
    const_arr = mu/(mu+1)**2
    const_y = const_arr[y._indices()[0]] * y._values()[:, None]
    for i in range(dim):
        for j in range(dim):
            if j > i:
                continue
            t = (-x[:,i,None] * x[:,j,None])
            k = (t * const_arr).sum(0) + (t[y._indices()[0]] * const_y).sum(0)
            hess_arr[i,j] = k
            hess_arr[j,i] = k
    return hess_arr
