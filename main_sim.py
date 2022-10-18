import os
import warnings
import torch
import numpy as np
from dataclasses import dataclass

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

SMOKE_TEST = os.environ.get("SMOKE_TEST")

NUM_FANTASIES = 128 if not SMOKE_TEST else 4

## load test functions and gp related
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.test_functions import Ackley, Branin, Bukin, Hartmann, Michalewicz, SixHumpCamel
from gpytorch.mlls import ExactMarginalLogLikelihood
from test_functions_add import *


import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HorseshoePrior
from gpytorch.constraints import Interval


## load acquisitions
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, qExpectedImprovement, qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from AcquisitionsNew import *
from torch.quasirandom import SobolEngine

### for Thompson sampling
import pykeops
import gpytorch
import gpytorch.settings as gpts
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from contextlib import ExitStack

## for utilities
from utils import *

## Specification for TurBO
#### Maintain the TurBO state
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 + math.fabs(state.best_value):
        state.success_counter += 1
        state.failurecbounter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state
        


## generate new batch via optimization
def generate_batch(
        curr_iter,
        model,
        X,
        Y,
        batch_size,
        acqf,
        options,
        num_restarts = 20,
        raw_samples = 512,
        post_processing_func = None, 
):
    assert acqf in ("ei", "ei_t", "adj_ei", "adj_ei_t", "poi", "poi_t", "adj_poi", "adj_poi_t", "ucb", "adj_ucb", "pr", "pr_t", "adj_pr", "adj_pr_t", "ts", "mes", "tr_ts", "tr_ei", "oskg")
    normal_X = options.get("normal_X", True)
    bounds = options.get("bounds")
    dim = X.shape[-1]
    if normal_X:
        bounds = torch.tensor([[0.] * dim, [1.] * dim])
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    else:
        assert torch.all(torch.hstack([torch.logical_and(X[:, i] >= bounds[0][i], X[:, i] <= bounds[1][i]) for i in range(dim)]))

    standardized_Y = options.get("standardized_Y", False)
    if standardized_Y:
        train_Y = standardize(Y)
    else:
        train_Y = Y
        
    opt_type = options.get("opt_type")
    threshold_value = options.get("threshold_value")
    seed = options.get("seed") + curr_iter * 25
    
    if "tr" in acqf:
        eval_type = acqf.split("_")[1]
        n_candidates = options.get("n_candidates", min(5000, max(2000, 200 * X.shape[-1])))
        state = options.get("state")
        
        X_double = X.to(dtype = torch.double)
        Y_double = train_Y.to(dtype = torch.double)
        likelihood_double = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model_double = SingleTaskGP(X_double, Y_double, covar_module=covar_module, likelihood=likelihood_double)
        # model_double = SingleTaskGP(X_double, Y_double)
        mll = ExactMarginalLogLikelihood(model_double.likelihood, model_double)
        fit_gpytorch_model(mll)
        
        # scale the TR to be proportional to the lengthscales
        x_center = X_double[Y_double.argmax(), :].clone()
        weights = model_double.covar_module.base_kernel.lengthscale.squeeze().detach()
        if dim > 1:
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        else:
            weights = torch.tensor(1.0)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if eval_type == "ts":
            sobol = SobolEngine(dim, scramble = True)
            pert = sobol.draw(n_candidates).to(dtype = torch.double, device = device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype = torch.double, device = device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim = 1) == 0)[0]
            if len(ind) > 0:
                mask[ind, torch.randint(0, dim - 1, size = (len(ind), ), device = device)] = 1
            else:
                mask = mask

            # create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model = model_double, replacement = False)
            with torch.no_grad():
                with manual_seed(seed):
                    X_next = thompson_sampling(X_cand, num_samples = batch_size)

        elif eval_type == "ei":
            EI = qExpectedImprovement(model_double, Y_double.max(), maximize = True)
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    EI,
                    bounds = torch.stack([tr_lb, tr_ub]),
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
            
    elif acqf == "ei":
        EI = ExpectedImprovement(model = model, best_f = train_Y.max(), maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = EI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = EI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func, 
                )
            

    elif acqf == "ei_t":
        EIt = ExpectedImprovement(model = model, best_f = threshold_value, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = EIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = EIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )

    elif acqf == "adj_ei":
        adjEI = AdjustedDistExpectedImprovement(model = model, best_f = train_Y.max(), X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjEI,	
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjEI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "adj_ei_t":
        adjEIt = AdjustedDistExpectedImprovement(model = model, best_f = threshold_value, X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjEIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjEIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "poi":
        PoI = ProbabilityOfImprovement(model = model, best_f = train_Y.max(), maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = PoI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = PoI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "poi_t":
        PoIt = ProbabilityOfImprovement(model = model, best_f = threshold_value, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = PoIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = PoIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "adj_poi":
        adjPoI = AdjustedDistProbabilityOfImprovement(model = model, best_f = train_Y.max(), X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjPoI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjPoI,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "adj_poi_t":
        adjPoIt = AdjustedDistProbabilityOfImprovement(model = model, best_f = threshold_value, X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjPoIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjPoIt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "ucb":
        beta = choose_beta(dim, curr_iter + 1)
        UCB = UpperConfidenceBound(model = model, beta = beta, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = UCB,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = UCB,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "adj_ucb":
        beta = choose_beta(dim, curr_iter + 1)
        adjUCB = AdjustedDistUpperConfidenceBound(model = model, beta = beta, X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjUCB,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjUCB,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "pr":
        PR = ProbabilityRatio(model = model, best_f = train_Y.max(), maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = PR,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = PR,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "pr_t":
        PRt = ProbabilityRatio(model = model, best_f = threshold_value, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = PRt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = PRt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "adj_pr":
        adjPR = AdjustedDistProbabilityRatio(model = model, best_f = train_Y.max(), X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjPR,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjPR,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "adj_pr_t":
        adjPRt = AdjustedDistProbabilityRatio(model = model, best_f = threshold_value, X_keep = X, maximize = True)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = adjPRt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = adjPRt,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "oskg":
        NUM_FANTASIES = options.get("num_fantasies", 128)
        qKG = qKnowledgeGradient(model, num_fantasies = NUM_FANTASIES)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = qKG,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = qKG,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )


    elif acqf == "ts":
        ts_sampler = options.get("ts_sampler", "cholesky")
        n_candidates = options.get("n_candidates", min(5000, max(2000, 200 * X.shape[-1])))
        use_keops = options.get("use_keops", False)
        assert ts_sampler in ("cholesky", "ciq", "rff", "lanczos")

        kernel_kwargs = {"nu": 2.5, "ard_num_dims": X.shape[-1]}
        if ts_sampler == "rff":
            base_kernel = RFFKernel(**kernel_kwargs, num_samples = 1024)
        else:
            base_kernel = (
                KmaternKernel(**kernel_kwargs) if use_keops else MaternKernel(**kernel_kwargs)
            )
        covar_module = ScaleKernel(base_kernel)

        X_double = X.to(dtype = torch.double)
        Y_double = Y.to(dtype = torch.double)
        mdl_double = SingleTaskGP(X_double, Y_double)
        mll_double = ExactMarginalLogLikelihood(mdl_double.likelihood, mdl_double)
        fit_gpytorch_model(mll_double)
        
        sobol = SobolEngine(X.shape[-1], scramble = True)
        X_cand = sobol.draw(n_candidates)

        if normal_X:
            X_cand = X_cand.to(dtype = torch.double, device = device)
        else:
            X_cand = unnormalize(X_cand, bounds).to(dtype = torch.double, device = device)
            
        with ExitStack() as es:
            if ts_sampler == "cholesky":
                es.enter_context(gpts.max_cholesky_size(float("Inf")))
            elif ts_sampler == "ciq":
                es.enter_context(gpts.fast_computations(covar_root_decomposition = True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(gpts.minres_tolerance(2e-3))
                es.enter_context(gpts.num_contour_quadrature(15))
            elif ts_sampler =="lanczos":
                es.enter_context(gpts.fast_computations(covar_root_decomposition = True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))
            elif ts_sampler == "rff":
                es.enter_context(gpts.fast_computations(covar_root_decomposition = True))

            thompson_sampling = MaxPosteriorSampling(model = mdl_double, replacement = False)
            X_next = thompson_sampling(X_cand, num_samples = batch_size)


    elif acqf == "mes":
        candidate_set = torch.rand(1000, dim, device = device, dtype = torch.double)
        # candidate_set = obj.bounds[0] + (obj.bounds[1] - obj.bounds[0]) * candidate_set

        X_double = X.to(dtype = torch.double)
        Y_double = Y.to(dtype = torch.double)
        mdl_double = SingleTaskGP(X_double, Y_double)
        mll_double = ExactMarginalLogLikelihood(mdl_double.likelihood, mdl_double)
        fit_gpytorch_model(mll_double)

        qMES = qMaxValueEntropy(mdl_double, candidate_set)
        if opt_type == "opt":
            with manual_seed(seed):
                X_next, acq_value = optimize_acqf(
                    acq_function = qMES,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                )
        elif opt_type == "sample":
            with manual_seed(seed):
                X_next, acq_value = sample_acqf(
                    acq_function = qMES,
                    bounds = bounds,
                    q = batch_size,
                    num_restarts = num_restarts,
                    raw_samples = raw_samples,
                    options = options,
                    post_processing_func = post_processing_func,
                )

    return X_next



    

## define the optimization / sampling loop
def AcqSampleBO(
        n_init,
        dfname,
        err_idx,
        negate,
        max_evals,
        batch_size,
        options,
        likelihood = None,
        covar_module = None, 
        seed = None,
        num_restarts = 10,
        raw_samples = 512,
        post_processing_func = None
):

    # initialize data
    obj, X, Y = initialize_func_data(n_init, dfname, err_idx, negate)
    best_observed_value = Y.max().item()
    Y = Y.view(len(Y), 1)

    options.update({"bounds": obj.bounds})

    if negate:
        for i in range(n_init):
            print(f"Iteration: {0}: X = {X[i].numpy()} | y = {- Y[i].numpy()}")
    else:
        for i in range(n_init):
            print(f"Iteration: {0}: X = {X[i].numpy()} | y = {Y[i].numpy()}")

    acqf = options.get("acqf", "ei")
    standardized_Y = options.get("standardized_Y", True)
    normal_X = options.get("normal_X", True)
    
    state = TurboState(obj.dim, batch_size=batch_size)
    options.update({"state": state})
    n_sugs = options.get("n_sugs", 1)

    #### define a helper function to unnormalize point and get evaluation                                                                                                                                  
    def eval_obj(x):
        if normal_X:
            return obj(unnormalize(x, obj.bounds))
        else:
            return obj(x)

    if likelihood is None:
        likelihood = GaussianLikelihood(noise_constraint = Interval(1e-8, 1e-3))

    if covar_module is None:
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims = obj.dim, lengthscale_constraint = Interval(0.005, 4.0))
        )

    N_get = 0
    screenXY = options.get("screenXY", None)
    
    while N_get < max_evals:
        # fit a GP model
        ## subset selection if required
        if screenXY is not None:
            train_X, train_Y = screenXY(X, Y)
        else:
            train_X = X
            train_Y = Y

        ## standardize Y if required
        if standardized_Y:
            train_Y = standardize(train_Y)

        if normal_X:
            train_X = normalize(train_X, obj.bounds)
            
        model = SingleTaskGP(train_X, train_Y, covar_module = covar_module, likelihood = likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        options.update({"X_keep": train_X})

        select_type = options.get("select_type", "ei")
        def sel_func(x):
            if select_type == "ei":
                acq = ExpectedImprovement(model, best_f = train_Y.max())
                res = acq(x).cpu().detach().item()
            elif select_type == "poi":
                acq = ProbabilityOfImprovement(model, best_f = train_Y.max())
                res = acq(x).cpu().detach().item()
            elif select_type == "ucb":
                beta = choose_beta(obj.dim, N_get)
                acq = UpperConfidenceBound(model, beta = beta)
                res = acq(x).cpu().detach().item()
            elif select_type == "postmu":
                res = model.posterior(x).mean
            return res
        
        options.update({"sel_func": sel_func})

        with gpytorch.settings.cholesky_jitter(1e-3):
            X_next = generate_batch(
                curr_iter = N_get, 
                model = model, 
                X = train_X,
                Y = train_Y,
                batch_size = batch_size,
                acqf = acqf,
                options = options,
                num_restarts = num_restarts,
                raw_samples = raw_samples,
                post_processing_func = post_processing_func,
            )

        X_next = X_next.view(-1, obj.dim)
        Y_next = torch.tensor(
            [eval_obj(x) for x in X_next], dtype = dtype, device = device
        ).unsqueeze(-1)

        if normal_X:
            ori_X_next = unnormalize(X_next, obj.bounds)
        else:
            ori_X_next = X_next
            
        X = torch.cat((X, ori_X_next), dim = 0)
        Y = torch.cat((Y, Y_next), dim = 0)
        
        # print current status
        if negate:
            for i in range(n_sugs):
                print(
                    f"Iteration: {N_get + i + 1}: X = {ori_X_next[i].numpy()} | y = {- Y_next[i].numpy()}"
                )
        else:
            for i in range(n_sugs):
                print(
                    f"Iteration: {N_get + i + 1}: X = ({ori_X_next[i].numpy()} | y = {Y_next[i].numpy()}"
                )
                
        N_get += n_sugs

    return X, Y


def sample_benchmark(dfn, acq, options, output_dir, n_sugs = 1, M = 30, num_restarts = 20, raw_samples = 512):
    if dfn in ("forrester", "gramacylee", "branin", "bukin", "cos25", "sixhump", "michalewicz_2"):
        n_init = 3
        if dfn == "forrester":
            options.update({"threshold_value": -0.5})
            budget = 30
        elif dfn == "gramacylee":
            options.update({"threshold_value": 0.})
            budget = 30
        elif dfn == "branin":
            options.update({"threshold_value": 1.})
            budget = 80
        elif dfn == "bukin":
            options.update({"threshold_value": 10.})
            budget = 150
        elif dfn == "cos25":
            options.update({"threshold_value": 0.1})
            budget = 250
        elif dfn == "sixhump":
            options.update({"threshold_value": -0.5})
            budget = 150
        elif dfn == "michalewicz_2":
            options.update({"threshold_value": -1.5})
            budget = 150
    elif dfn in ("ackley_6", "hartmann"):
        n_init = 7
        if dfn == "ackley_6":
            options.update({"threshold_value": 10.})
            budget = 250
        elif dfn == "hartmann":
            options.update({"threshold_value": -2.8})
            budget = 250
    elif dfn in ("ackley_10", "michalewicz_10"):
        n_init = 11
        if dfn == "ackley_10":
            options.update({"threshold_value": 15.})
            budget = 250
        elif dfn == "michalewicz_10":
            options.update({"threshold_value": -5.5})
            budget = 250
    else:
        n_init = 6
        options.update({"threshold_value": -3.5})
        budget = 250
        
    batch_size = 1
    res_acq = []
    negate = True
    print(dfn)
    print(acq)
    
    post_processing_func = post_process_func
    acqf = options.get("acqf", acq)
    standardized_Y = options.get("standardized_Y", True)
    normal_X = options.get("normal_X", True)
    
    for j in range(M):
        obj, X, Y = initialize_func_data(n_init, dfn, noise_err = None, negate = negate)
        Y = Y.view(-1, 1)
        print(f"Iteration {j}")
        options.update({"bounds": obj.bounds})
        
        for i in range(n_init):
            print(f"Iteration: {0}: X = {X[i].numpy()} | y = {- Y[i].numpy()}")
        
        state = TurboState(obj.dim, batch_size=batch_size)
        screen_criteria = options.get("screenXY", None)
        options.update({"state": state})
        
        def eval_obj(x):
            if normal_X:
                return obj(unnormalize(x, obj.bounds))
            else:
                return obj(x)
            
        N_get = 0
        seed = 1025
        options.update({"seed": seed * j + 1992})
        torch.manual_seed(seed * j + 1992)
        while N_get < budget:
            if screen_criteria is None:
                train_X = X
                train_Y = Y
            else:
                train_X, train_Y = ScreenXY(X, Y, screen_criteria)
            
            if normal_X:
                train_X = normalize(train_X, obj.bounds)
            
            best_observed_value = Y.max().item()
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            _ = fit_gpytorch_model(mll)
            options.update({"X_keep": train_X})
            sel_options = options.get("sel_options")
            def sel_func(x):
                sel_options.update({"best_f": best_observed_value})
                sel_options.update({"threshold_value": options.get("threshold_value")})
                sel_options.update({"X_keep": train_X})
                return selection_func(x, model, sel_options)
            
            options.update({"sel_func": sel_func})
            opt_type = options.get("opt_type")
            if opt_type == "random":
                if normal_X:
                    X_next = torch.rand(n_sugs, obj.dim)
                else:
                    X_next = unnormalize(torch.rand(n_sugs, obj.dim), obj.bounds)
            else:
                try:
                    X_next = generate_batch(N_get, model, train_X, train_Y, batch_size, acqf, options, num_restarts, raw_samples, post_processing_func)
                except:
                    with gpytorch.settings.cholesky_jitter(1e-3):
                        X_next = generate_batch(N_get, model, train_X, train_Y, batch_size, acqf, options, num_restarts, raw_samples, post_processing_func)
            
            X_next = X_next.view(-1, obj.dim)
            Y_next = torch.tensor([eval_obj(x) for x in X_next], dtype = dtype, device = device).unsqueeze(-1)
            
            # get back the original space
            if normal_X:
                ori_X_next = unnormalize(X_next, obj.bounds)
            else:
                ori_X_next = X_next
            # concatenate to the X dimension
            X = torch.cat((X, ori_X_next), dim = 0)
            Y = torch.cat((Y, Y_next), dim = 0)
            # print the result for each steps
            if negate:
                for k in range(n_sugs):
                    print(f"Iteration: {N_get + k + 1}: X = {ori_X_next[k].numpy()} | y = {- Y_next[k].numpy()}")
            else:
                for k in range(n_sugs):
                    print(f"Iteration: {N_get + k + 1}: X = {ori_X_next[k].numpy()} | y = {Y_next[k].numpy()}")
            
            N_get += n_sugs
            
        # save to numpy
        resX_np = X.cpu().detach().numpy()
        resY_np = Y.cpu().detach().numpy()
        if negate:
            res_np = np.hstack([resX_np, -resY_np])
        else:
            res_np = np.hstack([resX_np, resY_np])
        # append to the result list
        res_acq.append(res_np)
    # save to a file
    outputfile = os.path.join(output_dir, options.get("opt_type") + "_" + dfn + "_" + acq + "_" + options.get("gfn") + "_seed_" + str(options.get("seed")) + "_rep_" + str(M) + ".pickle")
    with open(outputfile, 'wb') as f:
        pickle.dump(res_acq, f)