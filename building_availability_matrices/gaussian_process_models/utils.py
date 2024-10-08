"""
Reference: https://peterroelants.github.io/posts/gaussian-process-kernels/ 
"""

import numpy as np
import tensorflow_probability as tfp
from statsmodels import api as sm
import logging

from building_availability_matrices.utils import CORR, NO_CLIENTS, UNCORR

logging.basicConfig(format="%(levelname)s:%(message)s")


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


# TODO: Make changes to function definition to take in arguments @daniel
def exp1(freq0, seq_len=100, n_clients=NO_CLIENTS):
    """
    Fetch covariance matrix as numpy array basis the kernel name
    """
    xlim = (-3, 3)
    x = np.expand_dims(np.linspace(*xlim, seq_len), 1)

    # # Kernel function:   k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
    # cov = (
    #     tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1.0, length_scale=0.5)
    #     .matrix(x, x)
    #     .numpy()
    # )

    # # Kernel function: k(x, y) = amplitude**2 * (1. + ||x - y|| ** 2 / (2 * scale_mixture_rate * length_scale**2)) ** -scale_mixture_rate

    # cov = (
    #     tfp.math.psd_kernels.RationalQuadratic(
    #         amplitude=0.2, length_scale=1.0, scale_mixture_rate=1.0
    #     )
    #     .matrix(x, x)
    #     .numpy()
    # )

    # # Kernel function : k(x, y) = amplitude**2 * exp(-2  / length_scale ** 2 * sum_k sin(pi * |x_k - y_k| / period) ** 2)

    # cov = (
    #     tfp.math.psd_kernels.ExpSinSquared(amplitude=1.0, length_scale=2.0, period=1.0)
    #     .matrix(x, x)
    #     .numpy()
    # )

    # Mixed kernel function
    def _get_avail_mat(
        periodic_length_scale=1.0,
        period=1.0,
        amplitude=1.0,
        local_length_scale=1.0,
        log_msg=None,
    ):
        logging.info(f"Preparing cov mat for {log_msg} case")
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

        y = np.random.multivariate_normal(
            mean=np.zeros(seq_len), cov=cov, size=n_clients
        )
        thresh = np.quantile(y, [0.5], method="higher", axis=1).T
        y_bin = (y >= thresh).astype(np.int8)

        for _ in range(n_clients):
            logging.info(
                f"Lag 1 auto-correlation value for client {_}: {sm.tsa.acf(y_bin[_,:], nlags=1)[-1]}"
            )

        return y_bin

    avail_mat = {
        CORR: _get_avail_mat(
            periodic_length_scale=1.0,
            period=1.0,
            amplitude=1.0,
            local_length_scale=1.0,
            log_msg=CORR,
        ),
        UNCORR: _get_avail_mat(
            periodic_length_scale=1.0,
            period=1.0,
            amplitude=1.0,
            local_length_scale=0.01,
            log_msg=CORR,
        ),
    }

    return avail_mat
