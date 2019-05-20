# Online KLIEP
An online change point detection package for python using the online version of KLIEP algorithm.

The detail of the algorithm is written in [Sequential Change-Point Detection
Based on Direct Density-Ratio Estimation](http://www.ms.k.u-tokyo.ac.jp/2012/CDKLIEP.pdf) by Sugiyama et al.

```python
from kliep import SequentialDensityRatioEstimator 

sdre = SequentialDensityRatioEstimator(y,n_rf,n_te,k)
for new_data in new_data_set:
    sdre(new_data)
```