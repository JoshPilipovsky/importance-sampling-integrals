# # # # # # # # # # # # # # # # # # # # # 

import sys
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from scipy.signal import hilbert
import seaborn as sns

# # # # # # # # # # # # # # # # # # # # # 

def hilbertTransform(f, grid, x, hilb_grid):
    eval_pt = (x - hilb_grid * h) / h
    return jnp.sum(jnp.interp(hilb_grid * h, grid, f) * jnp.sinc(eval_pt / 2) * jnp.sin(jnp.pi * eval_pt / 2))

def initializeGaussian(gridpoint, mu, sigma):
    return jnp.exp(1j * mu * gridpoint - 0.5 * gridpoint ** 2 * sigma ** 2)

# # # # # # # # # # # # # # # # # # # # # 

# # # vectorize function maps # # #
vfunc_IG = vmap(initializeGaussian, in_axes = (0, None, None))
vfunc_HT = vmap(hilbertTransform, in_axes = (None, None, 0, None))

# # # jit it up # # #
vfunc_IG_jit = jit(vfunc_IG)
vfunc_HT_jit = jit(vfunc_HT)

# # # # # # # # # # # # # # # # # # # # # 

# define cutoff and resolution
d, L = 25, 10001

# define HT resolution
h, M = 0.5, 5000

# create grid along each axis
grid = jnp.linspace(-d, d, L)

# create grid for HT
hilb_grid = jnp.linspace(-M, M, 2 * M + 1)

# generate input distribution
mu, sigma = 1, np.sqrt(2)

# Compute CF and CDF of initial distribution
CF = vfunc_IG_jit(grid, mu, sigma)

# Compute HT of CF
HT_CF = vfunc_HT_jit(CF, grid, grid, hilb_grid)

# plot to verify
plt.subplots()
plt.plot(grid, jnp.real(CF), '-k')
plt.plot(grid, jnp.imag(CF), '-b')
plt.show()
plt.subplots()
plt.plot(grid, jnp.real(HT_CF), '-k')
plt.plot(grid, jnp.imag(HT_CF), '-b')
plt.show()

