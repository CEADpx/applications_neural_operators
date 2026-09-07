"""
Residual-corrected Bayesian inference under distribution shift (hyperelasticity).

Fills the gap flagged in sec:bayesian ("Role of correction in Bayesian
inference") of the book chapter draft: the existing Bayesian results in
survey_work/applications/bayesian_inverse_problem_hyperelasticity use
uncorrected neural operators on an (approximately) in-distribution truth.
Here the truth is the OOD field from generate_ground_truth_ood.py, and for
each neural operator we run pCN-MCMC three ways:
    1. Reference: finite-element forward model (mcmc.surrogate_to_use=None)
    2. Uncorrected: raw neural-operator forward model
    3. Corrected: neural-operator prediction + one residual correction
       (HyperelasticityModel.residual_correct via CorrectedSurrogateModel)

Expectation, per eq:correction_error_bound and the topology-optimization
result in sec:topology: (2) should show visibly larger posterior-mean error
than (1), and (3) should sit close to (1), the same qualitative pattern
already demonstrated for the optimization application.

Run in the 'neuralopv2' conda environment. Requires
Results/ground_truth/data.npz from generate_ground_truth_ood.py first.

Usage:
    python run_bayesian_inversion_ood.py            # full run (article-scale)
    python run_bayesian_inversion_ood.py --quick     # smoke test (~1 min/chain)
"""
import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time

import numpy as np
import torch
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx.mesh import CellType
from mpi4py import MPI

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, "survey_work/problems/hyperelasticity")

sys.path.insert(0, os.path.join(ROOT, "src/plotting"))
sys.path.insert(0, os.path.join(ROOT, "src/data"))
sys.path.insert(0, os.path.join(ROOT, "src/prior"))
sys.path.insert(0, os.path.join(ROOT, "src/mcmc"))
sys.path.insert(0, os.path.join(ROOT, "src/nn"))
sys.path.insert(0, os.path.join(ROOT, "src/nn/deeponet"))
sys.path.insert(0, os.path.join(ROOT, "src/nn/pcanet"))
sys.path.insert(0, os.path.join(ROOT, "src/nn/fno"))
sys.path.insert(0, os.path.join(ROOT, "src/nn/mlp"))
sys.path.insert(0, MODEL_DIR)

from hyperelasticityModel import HyperelasticityModel
from mcmc import MCMC
from priorSampler import PriorSampler
from correctedSurrogateModel import CorrectedSurrogateModel
from nn_util import load_surrogate_model

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_PRIOR_AC = 0.005
TRAIN_PRIOR_CC = 0.2

# must match generate_ground_truth_ood.py -- this is the transform used LIVE
# by both the reference FE model and the surrogates' w->m mapping during
# inference (see module docstring there for why it's (100,1300) rather than
# the surrogates' actual (100,1000) training transform: it's what makes the
# frozen surrogates OOD while keeping ground truth/prior/reference mutually
# consistent, so the reference chain's identifiability isn't corrupted).
LOGN_SCALE = 100.0
LOGN_TRANSLATE = 1200.0
NX, NY = 50, 50
FE_ORDER = 1

# prior used for INFERENCE (pCN proposal + prior term in the posterior).
# Deliberately different from the training prior: same correlation length
# (sqrt(a/c) = 0.1581, unchanged) but ~25x the marginal variance (a,c both
# scaled down by ~5x). This is independent of the truth distribution shift
# already built into the ground truth (generate_ground_truth_ood.py).
#
# Calibrated via check_prior_shift.py (forward-only, no MCMC): at this
# setting, 10 draws from the shifted prior gave FE-converged states in all
# cases, with raw-surrogate state error averaging 2-5% (individual draws up
# to ~19%) -- a visible, moderate degradation, well below the article's
# transform-based OOD Case 1 (13-25%, tab:prediction_errors) and nowhere near
# Case 2 (50-70%, where the surrogates stop being useful at all). Shrinking
# correlation length alone (holding variance roughly fixed) was tested first
# and had almost no effect (errors stayed near 0%): the surrogates are
# sensitive to the amplitude/tail of m, not to its spatial frequency content.
INFERENCE_PRIOR_AC = 0.001
INFERENCE_PRIOR_CC = 0.04

PCN_BETA = 0.15


def build_model_and_prior():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, NX, NY, cell_type=CellType.triangle)
    Vm = functionspace(domain, ("Lagrange", FE_ORDER))
    Vu = functionspace(domain, ("Lagrange", FE_ORDER, (2,)))

    train_prior_sampler = PriorSampler(Vm, TRAIN_PRIOR_AC, TRAIN_PRIOR_CC, SEED)
    model = HyperelasticityModel(Vm, Vu, train_prior_sampler, LOGN_SCALE, LOGN_TRANSLATE, SEED)

    inference_prior = PriorSampler(Vm, INFERENCE_PRIOR_AC, INFERENCE_PRIOR_CC, SEED)

    return model, inference_prior


def load_surrogates(model):
    surrogate_models = {}
    for name, kwargs in [("DeepONet", {}), ("PCANet", {}), ("FNO", {"u_comps": 2})]:
        sm = load_surrogate_model(name, "Hyperelasticity", MODEL_DIR, model, **kwargs)
        if sm is not None:
            surrogate_models[name] = sm
            surrogate_models[name + "-corrected"] = CorrectedSurrogateModel(sm)
        else:
            print(f"{name} surrogate not found; skipping.")
    return surrogate_models


def posterior_mean_errors(mcmc):
    """Relative % L2 error in the posterior mean of w and m, evaluated with
    the reference FE model (independent of which model produced the chain),
    matching compute_sample_errors used elsewhere in this repository and
    Table "bayesian_errors" in the book chapter draft."""
    w_true = mcmc.data["w_true"]
    m_true = mcmc.data["m_true"]

    w_mean = mcmc.tracer.accepted_samples_mean_m
    m_mean = mcmc.model.transform_gaussian_pointwise(w_mean)

    err_w = 100.0 * np.linalg.norm(w_mean - w_true) / np.linalg.norm(w_true)
    err_m = 100.0 * np.linalg.norm(m_mean - m_true) / np.linalg.norm(m_true)
    return err_w, err_m


def run_chain(mcmc, surrogate_to_use, n_samples, n_burnin, savepath_base):
    mcmc.surrogate_to_use = surrogate_to_use
    tag = surrogate_to_use if surrogate_to_use is not None else "true_model"
    savepath = os.path.join(savepath_base, f"surrogate_{tag}")
    os.makedirs(savepath, exist_ok=True)

    t0 = time.time()
    mcmc.run(
        n_samples=n_samples,
        n_burnin=n_burnin,
        pcn_beta=PCN_BETA,
        sigma_noise=mcmc.sigma_noise,
        save_every=max(1, n_samples // 100),
        print_every=max(1, n_samples // 5),
        print_lvl=1,
        display_plot_every=n_samples + n_burnin + 1,  # effectively disable plotting
        savefilename="tracer",
        init_tracer=True,
        savepath=savepath + os.sep,
    )
    dt = time.time() - t0

    err_w, err_m = posterior_mean_errors(mcmc)
    print(f"[{tag}] wall time: {dt:.1f}s, error(w)={err_w:.3f}%, error(m)={err_m:.3f}%")
    return {"tag": tag, "wall_time_s": dt, "error_w_pct": err_w, "error_m_pct": err_m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke test (small sample count)")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_burnin", type=int, default=None)
    parser.add_argument(
        "--operators", nargs="+", default=["DeepONet", "PCANet", "FNO"],
        help="which neural operators to run (reference is always included)",
    )
    parser.add_argument(
        "--single", default=None,
        help="run only one chain (e.g. 'reference', 'DeepONet', 'DeepONet-corrected') "
             "and write Results/.../partial_<tag>.json instead of the aggregate "
             "summary.json. Lets independent chains be launched as separate "
             "parallel processes; merge_summaries.py combines the partials afterward.",
    )
    parser.add_argument(
        "--prior_ac", type=float, default=None,
        help="override INFERENCE_PRIOR_AC (module default calibrated for ~25x "
             "training variance at fixed correlation length); lets a calibration "
             "sweep vary prior severity from the command line without editing the file.",
    )
    parser.add_argument(
        "--prior_cc", type=float, default=None,
        help="override INFERENCE_PRIOR_CC (see --prior_ac).",
    )
    args = parser.parse_args()

    global INFERENCE_PRIOR_AC, INFERENCE_PRIOR_CC
    if args.prior_ac is not None:
        INFERENCE_PRIOR_AC = args.prior_ac
    if args.prior_cc is not None:
        INFERENCE_PRIOR_CC = args.prior_cc

    n_samples = args.n_samples or (50 if args.quick else 10000)
    n_burnin = args.n_burnin or (5 if args.quick else 500)

    results_dir = os.path.join(NOTEBOOK_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)

    gt_path = os.path.join(NOTEBOOK_DIR, "Results", "ground_truth", "data.npz")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"{gt_path} not found; run generate_ground_truth_ood.py first.")
    ground_truth_data = dict(np.load(gt_path))

    model, inference_prior = build_model_and_prior()
    surrogate_models = load_surrogates(model)

    u_obs_mean = np.mean(ground_truth_data["u_obs"])
    sigma_noise = 0.01 * u_obs_mean
    print(f"sigma_noise = {sigma_noise:.3e}")

    mcmc = MCMC(
        model,
        inference_prior,
        ground_truth_data,
        sigma_noise=sigma_noise,
        pcn_beta=PCN_BETA,
        surrogate_to_use=None,
        surrogate_models=surrogate_models,
        seed=SEED,
    )

    savepath_base = os.path.join(
        results_dir,
        f"mcmc_n_samples_{n_samples}_n_burnin_{n_burnin}_pcn_beta_{PCN_BETA:.3f}_sigma_{sigma_noise:.3e}"
        f"_priorac_{INFERENCE_PRIOR_AC:.3e}_priorcc_{INFERENCE_PRIOR_CC:.3e}",
    )

    if args.single is not None:
        surrogate_to_use = None if args.single == "reference" else args.single
        if surrogate_to_use is not None and surrogate_to_use not in surrogate_models:
            raise ValueError(
                f"--single {args.single!r} not available; have {sorted(surrogate_models)}"
            )
        result = run_chain(mcmc, surrogate_to_use, n_samples, n_burnin, savepath_base)
        os.makedirs(savepath_base, exist_ok=True)
        partial_path = os.path.join(savepath_base, f"partial_{result['tag']}.json")
        with open(partial_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved partial result to {partial_path}")
        return

    summary = []
    summary.append(run_chain(mcmc, None, n_samples, n_burnin, savepath_base))

    for op in args.operators:
        if op not in surrogate_models:
            print(f"{op} not available; skipping.")
            continue
        summary.append(run_chain(mcmc, op, n_samples, n_burnin, savepath_base))
        corrected_key = op + "-corrected"
        summary.append(run_chain(mcmc, corrected_key, n_samples, n_burnin, savepath_base))

    summary_path = os.path.join(savepath_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
