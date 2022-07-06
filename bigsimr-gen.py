from julia.api import Julia
jl = Julia(compiled_modules=False) 

from julia import Bigsimr as bs
from julia import Distributions as dist

import numpy as np

target_corr = bs.cor_randPD(3)
print(f'target_corr: {target_corr}')
binominal_dis = dist.Binomial(20, 0.2)
print(f'binominal distribution: {binominal_dis}')
margins = [dist.Binomial(20, 0.2), dist.Beta(2, 3), dist.LogNormal(3, 1)]
print(f'margins: {margins}')

adjusted_corr = bs.pearson_match(target_corr, margins)

x = bs.rvec(10, adjusted_corr, margins)
print(f'x type: {type}, value: {x}')
res = bs.cor(x, bs.Pearson)
print(f'corr {res}')