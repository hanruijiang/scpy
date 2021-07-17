# Scpy

**a statsmodels implement based on pytorch**

**can perform single-cell gene differential expression analysis 1000X faster than R package.**

## requirements

- pandas
- numpy
- statsmodels
- torch

## usage

### basic

```python
import scpy

scpy.fit(x, y, family='poisson', device='cpu')

scpy.fit_batch(X, Y, family='negative_binomial', device='cpu')

```

### fit parallel

```bash
python -m scpy demo.py demo.csv -d 0,1,2,3,4,5,6,7
```

## todo

- [ ] Currently only supports 'poisson' and 'negative_binomial', add more distributions.



