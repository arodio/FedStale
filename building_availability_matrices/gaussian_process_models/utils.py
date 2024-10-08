"""
Reference: https://peterroelants.github.io/posts/gaussian-process-kernels/ 

# Kernel function:   k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
cov = (
    tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1.0, length_scale=0.5)
    .matrix(x, x)
    .numpy()
)

# Kernel function: k(x, y) = amplitude**2 * (1. + ||x - y|| ** 2 / (2 * scale_mixture_rate * length_scale**2)) ** -scale_mixture_rate

cov = (
    tfp.math.psd_kernels.RationalQuadratic(
        amplitude=0.2, length_scale=1.0, scale_mixture_rate=1.0
    )
    .matrix(x, x)
    .numpy()
)

# Kernel function : k(x, y) = amplitude**2 * exp(-2  / length_scale ** 2 * sum_k sin(pi * |x_k - y_k| / period) ** 2)

cov = (
    tfp.math.psd_kernels.ExpSinSquared(amplitude=1.0, length_scale=2.0, period=1.0)
    .matrix(x, x)
    .numpy()
)

"""

import numpy as np
import tensorflow_probability as tfp

from utils import (
    CORR,
    NO_CLIENTS,
    UNCORR,
    CORR_FT,
    UNCORR_FT,
)


def get_local_periodic_kernel(
    periodic_length_scale, period, amplitude, local_length_scale
):
    """
    Composite kernel functions which is product of ExpSinSquared and Exponential Quadratic
    """
    periodic = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=amplitude, length_scale=periodic_length_scale, period=period
    )
    local = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=local_length_scale)
    return periodic * local


def get_cov_mat(
    x, periodic_length_scale=1.0, period=1.0, amplitude=1.0, local_length_scale=1.0
):
    cov = (
        get_local_periodic_kernel(
            periodic_length_scale=periodic_length_scale,
            period=period,
            amplitude=amplitude,
            local_length_scale=local_length_scale,
        )
        .matrix(x, x)
        .numpy()
    )
    return cov
