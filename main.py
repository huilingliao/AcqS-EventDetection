from main_sim import *

n_init = 3
dfn = "sixhump"
err_idx = None
negate = True
max_evals = 50
batch_size = 1

output_dir = "~/Downloads/"

options = {"opt_type": "sample",
           "threshold_value": 0.,
           "beta": 1.,
           "n_sugs": 1, 
           "return_best_only": True,
           "ts_sampler": "cholesky",
           "n_candidates": 1000,
           "alpha": 0.1,
           "batch_limit": 5,
           "init_batch_limit": 5,
           "nonnegative": True,
           "gfn": "exp",
           "M": 50,
           "L": 10,
           "epsilon": 1e-3,
           "burnin": 10,
           "n_gap": 5,
           "hmc_type": "hmc_wc",
           "x0_type": "qmc",
           "use_keops": False,
           "seed": 1025,
           "select_criteria": {"bws_type": "quantile", "bws_param": 0.5, "bwe_type": "quantile", "bwe_param": 0.5, "bwe_threshold": 0.1},
           "standardized_Y": False,
	   "normal_X": True,
           "screenXY": None,
           }




acq_list = ("ei", "ei_t", "adj_ei", "adj_ei_t", "poi", "poi_t", "adj_poi", "adj_poi_t", "ucb", "adj_ucb", "pr", "pr_t", "adj_pr", "adj_pr_t", "ts", "mes", "tr_ts", "tr_ei")
for acq in acq_list:
    options.update({"acqf": acq})
    print("acquisition: " + acq)
    resX, resY = AcqSampleBO(n_init, dfn, err_idx, negate, max_evals, batch_size, options)
    resX_np = resX.cpu().detach().numpy()
    resY_np = resY.cpu().detach().numpy()
    res_np = np.hstack([resX_np, resY_np])
    outputfile = dfn + "_" + acq + "_" + options.get("gfn") + "_seed_" + options.get("seed") + ".npy" 
    np.save(outputfile, res_np)
