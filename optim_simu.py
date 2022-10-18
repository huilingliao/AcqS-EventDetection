import os
import math
import warnings
import botorch
import torch
import pickle
import gpytorch

import numpy as np
from torch import Tensor

from main_sim import *

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from botorch.optim.optimize import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch
from botorch.utils.transforms import normalize, unnormalize
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.test_functions.base import BaseTestProblem


from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.analytic import *
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.utils import is_nonnegative

from botorch.optim.utils import columnwise_clamp
from botorch.optim.initializers import sample_points_around_best, initialize_q_batch_nonneg, initialize_q_batch

from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples

from botorch.logging import logger

from botorch.test_functions.synthetic import *
from test_functions_add import *

from HMC_pytorch.hmc import HMC, HMC_wc, unique
from scipy.stats.qmc import LatinHypercube

torch.random.manual_seed(1025)

def optim_benchmark(dfn, acq, output_dir, n_sugs = 1):
    if dfn in ("forrester", "gramacylee", "branin", "bukin", "cos25", "sixhump", "michalewicz_2"):
        n_init = 3
        if dfn == "forrester":
            threshold_value = -0.5
            budget = 30
        elif dfn == "gramacylee":
            threshold_value = 0.
            budget = 30
        elif dfn == "branin":
            threshold_value = 1.
            budget = 80
        elif dfn == "bukin":
            threshold_value = 10.
            budget = 150
        elif dfn == "cos25":
            threshold_value = 0.1
            budget = 250
        elif dfn == "sixhump":
            threshold_value = -0.5
            budget = 150
        elif dfn == "michalewicz_2":
            threshold_value = -1.5
            budget = 150
    elif dfn in ("ackley_6", "hartmann"):
        n_init = 7
        if dfn == "ackley_6":
            threshold_value = 5.
            budget = 250
        elif dfn == "hartmann":
            threshold_value = -2.8
            budget = 250
    elif dfn in ("ackley_10", "michalewicz_10"):
        n_init = 11
        if dfn == "ackley_10":
            threshold_value = 5.
            budget = 250
        elif dfn == "michalewicz_10":
            threshold_value = -5.5
            budget = 250
    else:
        n_init = 6
        threshold_value = -3.5
        budget = 250
        
    M = 30
    res_acq = []
    for j in range(M):
        obj, X, Y = initialize_func_data(n_init, dfn, noise_err = None, negate = True)
        print(f"Iteration {j}")
        for i in range(budget):
            train_X = normalize(X, obj.bounds)
            model = botorch.models.SingleTaskGP(train_X, Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_torch(mll, options={"maxiter": 3000, "lr": 0.01, "disp": False})
            except:
                try:
                    with gpytorch.settings.cholesky_jitter(1e-1):
                        fit_gpytorch_torch(mll, options={"maxiter": 3000, "lr": 0.01, "disp": False})
                except:
                    break
                    
            if acq == "ei":
                acquisition = botorch.acquisition.analytic.ExpectedImprovement(model, torch.max(Y), maximize=True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "ei_t":
                acquisition = ExpectedImprovement(model = model, best_f = threshold_value, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_ei":
                acquisition = AdjustedDistExpectedImprovement(model = model, best_f = Y.max(), X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_ei_t":
                acquisition = AdjustedDistExpectedImprovement(model = model, best_f = threshold_value, X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "poi":
                acquisition = ProbabilityOfImprovement(model = model, best_f = Y.max(), maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "poi_t":
                acquisition = ProbabilityOfImprovement(model = model, best_f = threshold_value, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_poi":
                acquisition = AdjustedDistProbabilityOfImprovement(model = model, best_f = Y.max(), X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_poi_t":
                acquisition = AdjustedDistProbabilityOfImprovement(model = model, best_f = threshold_value, X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "ucb":
                beta = choose_beta(obj.dim, i + 1)
                acquisition = UpperConfidenceBound(model = model, beta = beta, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_ucb":
                beta = choose_beta(obj.dim, i + 1)
                acquisition = AdjustedDistUpperConfidenceBound(model = model, beta = beta, X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "pr":
                acquisition = ProbabilityRatio(model = model, best_f = Y.max(), maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "pr_t":
                acquisition = ProbabilityRatio(model = model, best_f = threshold_value, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_pr":
                acquisition = AdjustedDistProbabilityRatio(model = model, best_f = Y.max(), X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "adj_pr_t":
                acquisition = AdjustedDistProbabilityRatio(model = model, best_f = threshold_value, X_keep = train_X, maximize = True)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "ts":
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

                X_double = train_X.to(dtype = torch.double)
                Y_double = Y.to(dtype = torch.double)
                mdl_double = SingleTaskGP(X_double, Y_double)
                mll_double = ExactMarginalLogLikelihood(mdl_double.likelihood, mdl_double)
                _ = fit_gpytorch_model(mll_double)

                sobol = SobolEngine(X.shape[-1], scramble = True)
                X_cand = sobol.draw(n_candidates)
                X_cand = X_cand.to(dtype = torch.double, device = device)

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
                    candidates = thompson_sampling(X_cand, num_samples = batch_size)
            elif acq == "mes":
                candidate_set = torch.rand(1000, obj.dim, device = device, dtype = torch.double)
                # candidate_set = obj.bounds[0] + (obj.bounds[1] - obj.bounds[0]) * candidate_set
                X_double = train_X.to(dtype = torch.double)
                Y_double = Y.to(dtype = torch.double)
                mdl_double = SingleTaskGP(X_double, Y_double)
                mll_double = ExactMarginalLogLikelihood(mdl_double.likelihood, mdl_double)
                fit_gpytorch_model(mll_double)
                acquisition = qMaxValueEntropy(mdl_double, candidate_set)
                candidates, acqui_value = optimize_acqf(acq_function=acquisition, 
                                                bounds = torch.tensor([[0.] * obj.dim, [1.] * obj.dim]),
                                                # bounds=test_fun.bounds,
                                                q=1,
                                                num_restarts=20,
                                                raw_samples=512,
                                                sequential=False,
                                                )
            elif acq == "tr_ts":
                n_candidates = options.get("n_candidates", min(5000, max(2000, 200 * train_X.shape[-1])))
                state = TurboState(obj.dim, batch_size=batch_size)
                X_double = train_X.to(dtype = torch.double)
                Y_double = Y.to(dtype = torch.double)
                likelihood_double = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
                )
                model_double = SingleTaskGP(X_double, Y_double, covar_module=covar_module, likelihood=likelihood_double)
                # model_double = SingleTaskGP(X_double, Y_double)
                mll = ExactMarginalLogLikelihood(model_double.likelihood, model_double)
                _ = fit_gpytorch_model(mll)

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
                        candidates = thompson_sampling(X_cand, num_samples = batch_size)
            elif acq == "tr_ei":
                n_candidates = options.get("n_candidates", min(5000, max(2000, 200 * train_X.shape[-1])))
                state = TurboState(obj.dim, batch_size=batch_size)
                X_double = train_X.to(dtype = torch.double)
                Y_double = Y.to(dtype = torch.double)
                likelihood_double = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
                )
                model_double = SingleTaskGP(X_double, Y_double, covar_module=covar_module, likelihood=likelihood_double)
                # model_double = SingleTaskGP(X_double, Y_double)
                mll = ExactMarginalLogLikelihood(model_double.likelihood, model_double)
                _ = fit_gpytorch_model(mll)

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
                EI = qExpectedImprovement(model_double, Y_double.max(), maximize = True)
                with manual_seed(seed):
                    candidates, acq_value = optimize_acqf(
                        EI,
                        bounds = torch.stack([tr_lb, tr_ub]),
                        q = batch_size,
                        num_restarts = num_restarts,
                        raw_samples = raw_samples,
                    )

            ori_X_next = unnormalize(candidates, obj.bounds)
            Y_next = obj(ori_X_next).view(-1, 1)
            X = torch.cat((X, ori_X_next), dim=0)
            Y = torch.cat((Y, Y_next), dim=0)
            for k in range(n_sugs):
                print(f"Iteration: {i + k + 1}: X = {ori_X_next[k].numpy()} | y = {-Y_next[k].numpy()}")
    # save to numpy
    resX_np = X.cpu().detach().numpy()
    resY_np = Y.cpu().detach().numpy()
    res_np = np.hstack([resX_np, -resY_np])
    # append to the result list
    res_acq.append(res_np)
    # save to a file
    outputfile = os.path.join(output_dir,"optim_" + dfn + "_" + acq + "_rep_30.pickle")
    with open(outputfile, 'wb') as f:
        pickle.dump(res_acq, f)



dfn_list = ("forrester")
acq_list = ("pr_t", "adj_pr", "adj_pr_t", "ts", "mes", "tr_ts", "tr_ei")

output_dir = "/Users/liao/Documents/BObystep/python_botorch/results/"
for dfn in dfn_list:
    for acq in acq_list:
        optim_benchmark(dfn, acq, output_dir)


dfn_list = ("gramacylee", "branin", "bukin", "cos25", "sixhump", "ackley_6", "ackley_10", 
            "hartmann", "michalewicz_2", "michalewicz_5", "michalewicz_10") 
acq_list = ("ei", "ei_t", "adj_ei", "adj_ei_t", "poi", "poi_t", "adj_poi", "adj_poi_t", "ucb", "adj_ucb", 
            "pr", "pr_t", "adj_pr", "adj_pr_t", "ts", "mes", "tr_ts", "tr_ei")


output_dir = "/Users/liao/Documents/BObystep/python_botorch/results/"
for dfn in dfn_list:
    for acq in acq_list:
        optim_benchmark(dfn, acq, output_dir)




