import torch
import math
from torch import Tensor
from typing import List, Optional, Tuple

from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.test_functions.base import BaseTestProblem

class Forrester(SyntheticTestFunction):
    r"""Forrester test function.

    1-dimensional function (evluated on `[0, 1]`):

        f(x) = (6x - 2)^2 x sin(12x - 4)

    f has two minimizers for its global minimum at 0.7572727 with f(x) = -6.02074
    """

    dim = 1
    _bounds = [(0, 1)]
    _optimal_value = -6.02074
    _optimizers = [(0.7572727)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.pow(6 * X - 2, 2) * torch.sin(12 * X - 4)


class Gramacylee(SyntheticTestFunction):
    r"""Gramacylee test function.

    1-dimensional function (evaluated on `[0.5,2.5]`)

        f(x) = sin(10 * pi * x) / (2 * x) + (x - 1)^4

    f has multiple minimizers for its global minimum at 0.5484848 with f(x) = -0.8690084
    """

    dim = 1
    _bounds = [(0.5, 2.5)]
    _optimal_value = -0.8690084
    _optimizers = [(0.5484848)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return 0.5 * torch.sin(10 * math.pi * X) / X + torch.pow(X - 1, 4)


class Cos25(SyntheticTestFunction):
    r"""Cos25 test function.

    2-dimensional function (evaluated on `[-5 * math.pi, 5 * math.pi]`)

        f(x) = -cos(x[1]) - cos(x[2]) + 2
    
    f has 25 minimizers for its global minimum at (−4pi, −2pi, 0, 2pi, 4pi) x (−4pi, −2pi, 0, 2pi, 4pi)
    with f(x) = 0
    """

    dim = 2
    _bounds = [(-5 * math.pi, 5 * math.pi), (-5 * math.pi, 5 * math.pi)]
    _optimal_value = 0
    _optimizers = [(x, y) for x in [-4 * math.pi, -2 * math.pi, 0.0, 2 * math.pi, 4 * math.pi] for y in [-4 * math.pi, -2 * math.pi, 0.0, 2 * math.pi, 4 * math.pi]]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return - torch.cos(X[:, 0]) - torch.sin(X[:, 1]) + 2



