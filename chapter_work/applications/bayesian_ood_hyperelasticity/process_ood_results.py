"""
Post-process the locked-in OOD Bayesian MCMC results (7 chains: reference +
raw/corrected DeepONet, PCANet, FNO), mirroring the plotting pipeline in
survey_work/applications/bayesian_inverse_problem_hyperelasticity/
prcessed_results/processResults.ipynb, but pointed at this experiment's
ground truth and MCMC run directory, and extended to include the corrected
surrogates (CorrectedSurrogateModel), which the original notebook predates.

Run in the 'neuralopv2' conda environment from this directory:
    python process_ood_results.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib
matplotlib.use("Agg")  # no interactive windows; plt.show() below becomes a no-op
import matplotlib.pyplot as plt
import numpy as np
import torch
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx.mesh import CellType
from mpi4py import MPI

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, "survey_work/problems/hyperelasticity")

sys.path.insert(0, MODEL_DIR)

from neural_operators.prior.priorSampler import PriorSampler
from neural_operators.mcmc.mcmc import MCMC
from neural_operators.mcmc.mcmc_postprocess import plot_mcmc_tracer_results
from neural_operators.mcmc.correctedSurrogateModel import CorrectedSurrogateModel
from neural_operators.nn.nn_util import load_surrogate_model
from neural_operators.plotting.plot_mix_collection import get_default_plot_mix_collection_data, plot_mix_collection
from hyperelasticityModel import HyperelasticityModel

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# --- locked-in experiment configuration (must match run_bayesian_inversion_ood.py) ---
DATA_PREFIX = "Hyperelasticity"
U_COMPS = 2

prior_ac = 0.0022361
prior_cc = 0.089443
prior_logn_scale = 100.0
prior_logn_translate = 1200.0

n_samples = 10000
n_burnin = 500
pcn_beta = 0.15

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results")
GROUND_TRUTH_DIR = os.path.join(RESULTS_DIR, "ground_truth")
MCMC_RUN_DIR = os.path.join(
    RESULTS_DIR,
    "mcmc_n_samples_10000_n_burnin_500_pcn_beta_0.150_sigma_9.800e-04"
    "_priorac_2.236e-03_priorcc_8.944e-02",
)
OUTPUT_DIR = os.path.join(NOTEBOOK_DIR, "prcessed_results", "Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_CONFIGS = [
    ("True", os.path.join(MCMC_RUN_DIR, "surrogate_true_model"), None),
    ("DeepONet", os.path.join(MCMC_RUN_DIR, "surrogate_DeepONet"), "DeepONet"),
    ("DeepONet-corrected", os.path.join(MCMC_RUN_DIR, "surrogate_DeepONet-corrected"), "DeepONet-corrected"),
    ("PCANet", os.path.join(MCMC_RUN_DIR, "surrogate_PCANet"), "PCANet"),
    ("PCANet-corrected", os.path.join(MCMC_RUN_DIR, "surrogate_PCANet-corrected"), "PCANet-corrected"),
    ("FNO", os.path.join(MCMC_RUN_DIR, "surrogate_FNO"), "FNO"),
    ("FNO-corrected", os.path.join(MCMC_RUN_DIR, "surrogate_FNO-corrected"), "FNO-corrected"),
]

MCMC_PP_PARAMS = {
    "curve_plot": {"fs": 20, "lw": 3, "figsize": (6, 4)},
    "field_plot": {
        "fs": 24,
        "y_sup_title": 1.075,
        "figsize": (20, 12),
        "ttl_pad": 10,
        "title_x": [0.6, 0.65, 0.65, 0.6],
        "u_vec_plot": True,
    },
}

FIELD_SUPTITLE = (
    r"Ground truth, posterior sample, and posterior mean $(w, m, u(m), u_{obs})$"
)


def main():
    nx, ny = 50, 50
    fe_order = 1

    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=CellType.triangle)
    Vm = functionspace(domain, ("Lagrange", fe_order))
    Vu = functionspace(domain, ("Lagrange", fe_order, (2,)))

    prior_sampler = PriorSampler(Vm, prior_ac, prior_cc, seed)
    model = HyperelasticityModel(
        Vm, Vu, prior_sampler, prior_logn_scale, prior_logn_translate, seed
    )
    nodes = model.m_nodes

    surrogate_models = {}
    for name, kwargs in [("DeepONet", {}), ("PCANet", {}), ("FNO", {"u_comps": U_COMPS})]:
        sm = load_surrogate_model(name, DATA_PREFIX, MODEL_DIR, model, **kwargs)
        if sm is not None:
            surrogate_models[name] = sm
            surrogate_models[name + "-corrected"] = CorrectedSurrogateModel(sm)
        else:
            print(f"{name} surrogate not found; skipping.")

    ground_truth_data = dict(np.load(os.path.join(GROUND_TRUTH_DIR, "data.npz")))
    num_grid_x = int(ground_truth_data["num_grid_x"])
    num_grid_y = int(ground_truth_data["num_grid_y"])
    w_true = ground_truth_data["w_true"]
    m_true = ground_truth_data["m_true"]
    u_true = ground_truth_data["u_true"]
    x_obs = ground_truth_data["x_obs"]
    u_obs = ground_truth_data["u_obs"]

    u_obs_mean, u_obs_std = np.mean(u_obs), np.std(u_obs)
    sigma_noise = 0.01 * u_obs_mean
    print(f"u_obs_mean = {u_obs_mean:.3e}, u_obs_std = {u_obs_std:.3e}, sigma_noise = {sigma_noise:.3e}")

    # --- ground truth panel: w_true, m_true, u_true, observed u ---
    ground_truth_fig = os.path.join(OUTPUT_DIR, "true_and_obs_w_m_u.png")
    plot_data = get_default_plot_mix_collection_data(
        rows=1,
        cols=4,
        nodes=nodes,
        nodes_point_plot=x_obs,
        figsize=(26, 6),
        fs=25,
        sup_title=rf"OOD ground truth $(w, m, u(m))$ and observed $u$ on {num_grid_x}x{num_grid_y} grid",
        y_sup_title=1.075,
        savefilename=ground_truth_fig,
        u=[[w_true, m_true, u_true, u_obs]],
        cmap=[["magma", "jet", "jet", "copper"]],
        title=[[
            r"$w_{true}$",
            r"$m_{true} = \alpha_m\, \exp(w_{true}) + \beta_m$",
            r"$u_{true} = F(m_{true})$",
            r"$\mathrm{g}_{true} = \bar{\mathbf{B}}(u_{true})$",
        ]],
        is_vec=[[False, False, True, True]],
        add_disp=[[False, False, True, True]],
        plot_type=[["field", "field", "field", "point"]],
    )
    plot_mix_collection(plot_data)
    print(f"Saved {ground_truth_fig}")

    # --- MCMC object reused for post-processing (matches run_bayesian_inversion_ood.py) ---
    mcmc = MCMC(
        model,
        prior_sampler,
        ground_truth_data,
        sigma_noise=sigma_noise,
        pcn_beta=pcn_beta,
        surrogate_to_use=None,
        surrogate_models=surrogate_models,
        seed=seed,
    )

    def process_run(tag, run_dir, surrogate_name):
        tracer_path = os.path.join(run_dir, "tracer.pkl")
        if not os.path.isfile(tracer_path):
            print(f"Skipping {tag}: missing {tracer_path}")
            return None

        mcmc.surrogate_to_use = surrogate_name
        field_suptitle = f"{FIELD_SUPTITLE} using {tag}"
        plot_surrogate_forward = surrogate_name is not None

        tag_out_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(tag_out_dir, exist_ok=True)

        tracer = plot_mcmc_tracer_results(
            run_dir,
            mcmc,
            MCMC_PP_PARAMS,
            field_suptitle,
            tag=tag,
            plot_surrogate_forward=plot_surrogate_forward,
            save_figures=True,
        )
        print(
            f"{tag}: acceptances={tracer.acceptances}, "
            f"final acceptance rate={tracer.acceptance_rate[-1]:.3e}"
        )
        return tracer

    only_tags = os.environ.get("ONLY_TAGS")
    run_configs = RUN_CONFIGS
    if only_tags:
        wanted = set(only_tags.split(","))
        run_configs = [c for c in RUN_CONFIGS if c[0] in wanted]

    tracers = {}
    for tag, run_dir, surrogate_name in run_configs:
        print(f"\n=== {tag} ===")
        tracer = process_run(tag, run_dir, surrogate_name)
        if tracer is not None:
            tracers[tag] = tracer

    print(f"\nProcessed {len(tracers)}/{len(RUN_CONFIGS)} runs.")
    print(f"Per-run figures (cost, acceptance rate, true/surrogate-forward field panels) "
          f"were saved directly in each run's own directory under {MCMC_RUN_DIR}.")
    print(f"Ground-truth panel saved to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
