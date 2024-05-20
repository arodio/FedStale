import torch
import torch.nn as nn

from copy import deepcopy


def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    r"""computes the average of learners and store it into target_learner

    Parameters
    ----------
    learners: List[Learner]

    target_learner: Learner

    weights: 1-D torch.tensor
        tensor of the same size as learners, having values between 0 and 1, and summing to 1,
        if not provided, uniform weights are used

    average_params: bool
        if set to true the parameters are averaged; default is True

    average_gradients: bool
        if set to true the gradient are averaged; default is False

    Returns
    -------
        None
    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    param_tensors = []
    grad_tensors = []

    for learner in learners:
        if average_params:
            param_tensors.append(deepcopy(learner.get_param_tensor()))

        if average_gradients:
            grad_tensors.append(deepcopy(learner.get_grad_tensor()))

    if average_params:
        param_tensors = torch.stack(param_tensors)
        average_params_tensor = weights @ param_tensors
        target_learner.set_param_tensor(average_params_tensor)

    if average_gradients:
        grad_tensors = torch.stack(grad_tensors)
        average_grads_tensor = weights @ grad_tensors
        target_learner.set_grad_tensor(average_grads_tensor)


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def copy_gradient(target, source):
    """
    Copy param.grad.data from source to target
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.grad.data = source_param.grad.data.clone()


def partial_average(learners, average_learner, alpha):
    r"""
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data =\
                    (1-alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data


def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())


def simplex_projection(v, s=1):
    r"""
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w
