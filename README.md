# Welcome to elnetpy!

![CI](https://github.com/PabloRMira/elnetpy/workflows/CI/badge.svg)

![Cov](https://github.com/PabloRMira/elnetpy/blob/master/img/coverage.svg)

A Python package to compute the elastic net algorithm, boosted by an own C++ implementation.

> Work in progress!

## Promising performance

Our package shows promising performance in comparison to the [Python port of the Fortran code](https://github.com/civisanalytics/python-glmnet/tree/master/glmnet) for the `glmnet` package. However, our package does not offer as many features as `glmnet` does.

Output of timings via pytest-benchmark:

```
tests/performance_tests/test_elnetpy_glmnet.py ......                    [100%]


---------------------------------------------------------------------------------------- benchmark: 6 tests ---------------------------------------------------------------------------------------
Name (time in ms)                 Min                 Max                Mean             StdDev              Median                IQR            Outliers       OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_linear_elnetpy[1]         1.3159 (1.0)        6.5030 (1.53)       1.5182 (1.0)       0.3374 (1.32)       1.4534 (1.0)       0.1195 (1.0)         30;48  658.6832 (1.0)         506           1
test_linear_elnetpy[0.5]       1.3345 (1.01)       4.2456 (1.0)        1.5505 (1.02)      0.2566 (1.0)        1.4802 (1.02)      0.1232 (1.03)        52;59  644.9656 (0.98)        461           1
test_linear_elnetpy[0]         1.3483 (1.02)      12.5787 (2.96)       2.0079 (1.32)      0.7661 (2.99)       1.8525 (1.27)      0.7987 (6.68)        36;12  498.0347 (0.76)        468           1
test_linear_glmnet[0.5]      109.4735 (83.19)    266.6773 (62.81)    163.0891 (107.42)   56.6493 (220.78)   145.6602 (100.22)   90.6115 (758.16)        1;0    6.1316 (0.01)          9           1
test_linear_glmnet[1]        147.7671 (112.29)   304.9664 (71.83)    236.9216 (156.06)   62.1737 (242.31)   258.0482 (177.55)   91.6640 (766.97)        2;0    4.2208 (0.01)          5           1
test_linear_glmnet[0]        164.6977 (125.16)   186.6691 (43.97)    169.9443 (111.94)    9.3810 (36.56)    165.8382 (114.10)    6.3995 (53.55)         1;1    5.8843 (0.01)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## References

Paper to `glmnet`: https://web.stanford.edu/~hastie/Papers/glmnet.pdf

`glmnet` (read-only) GitHub repo: https://github.com/cran/glmnet
