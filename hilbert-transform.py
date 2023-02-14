# # # # # # # # # # # # # # # # # # # # # 

import sys
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from scipy.signal import hilbert
import seaborn as sns

# # # # # # # # # # # # # # # # # # # # # 

def hilbertTransform_sinc(f, grid, x, hilb_grid):
    eval_pt = (x - hilb_grid * h) / h
    return jnp.sum(jnp.interp(hilb_grid * h, grid, f) * jnp.sinc(eval_pt / 2) * jnp.sin(jnp.pi * eval_pt / 2))

def computeExpectation(t, tau_grid, mu, sigma):
    PDF = vfunc_PDF_freq_jit(tau_grid, mu, sigma)
    return (1 / L) * jnp.sum((evalGaussian(t - tau_grid, mu, sigma) - evalGaussian(t + tau_grid, mu, sigma)) / (tau_grid * PDF))

def hilbertTransform_sampling(t, CF, PDF):
    (1 / jnp.pi) * computeExpectation(t, CF, PDF)

# def hilbertTransform_sinc(f, x):

# def frequencyPDF(CF, t):

def evalGaussian(t, mu, sigma):
    return jnp.exp(1j * mu * t - 0.5 * t ** 2 * sigma ** 2)

def absCF(t, mu, sigma):
    CF_t = evalGaussian(t, mu, sigma)
    return abs(CF_t)

def gradabsCF(t, mu, sigma):
    return grad(absCF)(t, mu, sigma)

def freqPDF(t, mu, sigma):
    return abs(gradabsCF(t, mu, sigma))

# # # # # # # # # # # # # # # # # # # # # 

# # # vectorize function maps # # #
# vfunc_IG = vmap(initializeGaussian, in_axes = (0, None, None))
# vfunc_HT_sinc = vmap(hilbertTransform_sinc, in_axes = (None, None, 0, None))
vfunc_PDF_freq    = vmap(freqPDF, in_axes = (0, None, None))
vfunc_grad_abs_CF = vmap(gradabsCF, in_axes = (0, None, None))
vfunc_abs_CF      = vmap(absCF, in_axes = (0, None, None))

# # # # jit it up # # #
# vfunc_IG_jit = jit(vfunc_IG)
# vfunc_HT_sinc_jit = jit(vfunc_HT_sinc)
vfunc_PDF_freq_jit    = jit(vfunc_PDF_freq)
vfunc_grad_abs_CF_jit = jit(vfunc_grad_abs_CF)
vfunc_abs_CF_jit      = jit(vfunc_abs_CF)

# # # # # # # # # # # # # # # # # # # # # 

# generate input distribution
mu, sigma = 1, np.sqrt(2)
grid = jnp.linspace(-5, 5, 1001)

t = 1.25
# M = 5
# eps = 1E-05
# L = 10000
# tau_grid = jnp.linspace(eps, M, L)
# E = computeExpectation(t, tau_grid, mu, sigma)

abs_CF_t      = absCF(t, mu, sigma)
abs_CF        = vfunc_abs_CF_jit(grid, mu, sigma)

grad_abs_CF_t = gradabsCF(t, mu, sigma)
grad_abs_CF   = vfunc_grad_abs_CF_jit(grid, mu, sigma)

PDF_freq_t    = freqPDF(t, mu, sigma)
PDF_freq      = vfunc_PDF_freq_jit(grid, mu, sigma)

plt.subplots()
plt.plot(grid, abs_CF, '-k')
plt.plot(grid, grad_abs_CF, '-b')
plt.plot(grid, PDF_freq, '-r')
plt.show()

temp = 1










# define cutoff and resolution
d, L = 5, 10001

# define HT resolution
h, M = 0.5, 5000

# create grid along each axis
grid = jnp.linspace(-d, d, L)

# create grid for HT
hilb_grid = jnp.linspace(-M, M, 2 * M + 1)



# Compute CF and CDF of initial distribution
CF = vfunc_IG_jit(grid, mu, sigma)
deriv_abs_CF = grad(absCF)(grid, mu, sigma)

# Compute HT of CF
HT_CF = vfunc_HT_sinc_jit(CF, grid, grid, hilb_grid)

# plot to verify
plt.subplots()
plt.plot(grid, jnp.real(CF), '-k')
plt.plot(grid, jnp.imag(CF), '-b')
plt.plot(grid, abs_CF, '-r')
plt.show()
plt.subplots()
plt.plot(grid, jnp.real(HT_CF), '-k')
plt.plot(grid, jnp.imag(HT_CF), '-b')
plt.show()

