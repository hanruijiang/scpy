# Scpy

**a statsmodels implement based on pytorch**

**can perform single-cell gene differential expression analysis 1000X faster than R package.**

## requirements

- pandas
- statsmodels
- torch

## usage

### basic

```python
import scpy

scpy.fit(x, y, family='poisson', device='cpu', **kwargs)

scpy.fit_batch(X, Y, family='negative_binomial', device='cpu', **kwargs)

```
Support specifying *method*, *maxiter* and other parameters in [statsmodels optimizers](https://www.statsmodels.org/stable/optimization.html)

### parallel

```bash
python -m scpy demo.py demo.csv -d 0,1,2,3,4,5,6,7
```
>*demo.py*
>```python
>args = {
>  'X': X,
>  'Y': Y,
>  'family': 'negative_binomial',
>  'xnames': xnames,
>  'yname': yname,
>  'kwargs': {}
>}
>```

## todo

- [ ] Currently only supports 'poisson' and 'negative_binomial', add more distributions.
- [ ] Fix bug on Hessian matrix inversion.
- [ ] Automatically cache x and y in fit_batch.


