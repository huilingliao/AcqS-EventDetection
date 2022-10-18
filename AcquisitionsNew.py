import math
import torch
from torch import Tensor
from typing import Optional, Union

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform

from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

from torch.distributions import Normal

## probability ratio (PR)
class ProbabilityRatio(AnalyticAcquisitionFunction):
    r"""Single-outcome Probability Ratio.

    Probability ratio over the current best observed value, computed 
    using the analytic formula under a normal posterior distribution. 
    Requires the posterior to be Gaussian. 
    The model must be single-outcome.

    `PR(x) = P(y >= best_f) / P(y < best_f), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PR = ProbabilityRatio(model, best_f=0.1)
        >>> pr = PR(test_X)
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            maximize: bool = True,
    ) -> None:
        r""" Single-outcome analytic Probability Ratio

        Args:
            model: A fitted single-outcome model
            best_f:  Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability Ratio on the candidate set X.

        Args: 
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(X = X)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        ccdf = (1 - ucdf).clamp_min(1e-9)
        return ucdf / ccdf

    
## adjusted EI
class AdjustedDistExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Adjusted Distance Expected Improvement (analytic)
    
    Compute the classic Expected Improvement over the current best observed value, 
    using the analytic formula for a Normal posterior distribution. 
    Adjust the EI value by the distance between the candidate point and existing points.

    `adjtEI(x) = E(max(y - best_f, 0)) x adjT, y ~ f(x)`
    `adjT = sqrt(prod(norm(x - X, "2"))`

    Example: 
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> adjTEI = AdjustedDistExpectedImprovement(model, best_f, X)
        >>> adjtei = adjTEI(test_X)
    
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            X_keep: Tensor,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Adjusted Distance Expected Improvement (analytic)

        Args: 
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            X: Either an array, or a `b`-dim Tensor representing the observed points so far.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.X_keep = torch.as_tensor(X_keep)
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Adjusted Distance Expected Improvement on the candidate set X. 

        Args: 
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.
        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(X=X)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        adjT = torch.as_tensor([torch.sqrt(torch.prod(torch.norm(xx.expand_as(self.X_keep) - self.X_keep, dim = 1))) for xx in X]).view(view_shape)
        return adjT * ei

## adjusted PoI
class AdjustedDistProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome adjusted distance probability of improvement (analytic)
    
    Probabiliyt of improvement over the current best observed value, 
    computed using the analytic formula under a normal posterior distribution. 
    Adjust by the distance to current observed points. 

    `adjTPI(x) = adjT x P(y >= best_f), y ~ f(x)`

    Example: 
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> adjTPI = AdjustedDistProbabilityOfImprovement(model, best_f, X_keep)
        >>> adjtpi = adjTPI(test_X)
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            X_keep: Tensor,
            maximize: bool = True,
    ) -> None:
        r"""

        Args:
            model: A fitted single-outcome model
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing 
                the best function value observed so far
            X_keep: Tensor, currently observed points
            posterior_transform: A PosteriorTransform. If using a multi-output model, 
                a PosteriorTransform that transforms the multi-output posterior into 
                a single-output posterior is required
            maximize: If True, consider the problem a maximization problem
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.X_keep = X_keep
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(X=X)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        adjT = torch.as_tensor([torch.sqrt(torch.prod(torch.norm(xx.expand_as(self.X_keep) - self.X_keep, dim = 1))) for xx in X]).view(batch_shape)
        return normal.cdf(u) * adjT
        

## adjusted UCB
class AdjustedDistUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome adjusted distance upper confidence bound (analytic) 
    
    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `adjTUCB(x) = adjT x (mu(x) + sqrt(beta) * sigma(x))`, 

    where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    Adjust by the distance to current points. 

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> adjTUCB = AdjustedDistUpperConfidenceBound(model, beta=0.2)
        >>> adjtucb = adjTUCB(test_X)
    """
    
    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            X_keep: Tensor,
            maximize: bool = True, 
    ) -> None:
        r"""

        Args: 
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            X_keep: Tensor, currently observed points
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.X_keep = X_keep
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.

        """
        self.beta = self.beta.to(X)
        posterior = self.model.posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        adjT = torch.as_tensor([torch.sqrt(torch.prod(torch.norm(xx.expand_as(self.X_keep) - self.X_keep, dim = 1))) for xx in X]).view(batch_shape)
        if self.maximize:
            return adjT * (mean + delta)
        else:
            return adjT * (-mean + delta)

        
## adjuster probability ratio
class AdjustedDistProbabilityRatio(AnalyticAcquisitionFunction):
    r"""Single-outcome adjusted probability ratio (analytic)

    Probability of improvement over the current best observed value, 
    using the analytic formula under a Normal posterior distribution.
    Consider the probability ratio instead the PI itself.
    Adjust by the distance to currently observed points.

    `adjTPR(x) = adjT x P(y >= best_f) / P(y < best_f), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> adjTPR(x) = AdjustedDistProbabilityRatio(model, best_f)
        >>> adjtpr = adjTPR(test_X)

    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            X_keep: Tensor,
            maximize: bool = True,
    ) -> None:
        r"""

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            X_keep: A Tensor, storing currently observed points
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.X_keep = X_keep
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)


    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        adjT = torch.as_tensor([torch.sqrt(torch.prod(torch.norm(xx.expand_as(self.X_keep) - self.X_keep, dim = 1))) for xx in X]).view(batch_shape)
        ucdf = normal.cdf(u)
        ccdf = (1 - ucdf).clamp_min(1e-9)
        return adjT * ucdf / ccdf

    
