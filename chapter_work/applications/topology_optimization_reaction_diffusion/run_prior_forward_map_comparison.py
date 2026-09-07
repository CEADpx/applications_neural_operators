"""
Forward-map comparison on fresh draws from the prior used to generate
DeepONet's training data (a=0.005, c=0.2, m=0.25*exp(w)), not the
optimization results. For each sample m: true FE solve, DeepONet-only
prediction, DeepONet + one-Newton-step correction.

Montage style matches survey_work/problems/{poisson,linear_elasticity,
hyperelasticity}/compare_nops/compareNeuralOperators.ipynb: rows are
quantities (Input m, True, DeepONet, DeepONet+Corr), columns are samples,
cmap='jet' throughout, error titles "$e = X.XX\\%$" (montage_util.py copies
that notebook's helpers).

Run in the 'neuralopv2' conda environment:
    python run_prior_forward_map_comparison.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolfinx.fem import functionspace
from PIL import Image

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, "..", "..", ".."))
PROBLEM_DIR = os.path.join(ROOT, "chapter_work", "problems", "reaction_diffusion")

sys.path.insert(0, os.path.join(ROOT, "src", "plotting"))
sys.path.insert(0, os.path.join(ROOT, "src", "prior"))
sys.path.insert(0, os.path.join(ROOT, "src", "pde"))
sys.path.insert(0, os.path.join(ROOT, "src", "data"))
sys.path.insert(0, os.path.join(ROOT, "src", "mcmc"))
sys.path.insert(0, os.path.join(ROOT, "src", "nn"))
sys.path.insert(0, os.path.join(ROOT, "src", "nn", "deeponet"))
sys.path.insert(0, os.path.join(ROOT, "src", "nn", "pcanet"))
sys.path.insert(0, os.path.join(ROOT, "src", "nn", "fno"))
sys.path.insert(0, os.path.join(ROOT, "src", "nn", "mlp"))
sys.path.insert(0, PROBLEM_DIR)
sys.path.insert(0, NOTEBOOK_DIR)

from field_plot import quick_field_plot
from mesh_setup import build_mesh
from priorSampler import PriorSampler
from reactionDiffusionModel import ReactionDiffusionModel
from nn_util import load_surrogate_model
from montage_util import save_image_montage, format_pct_error, format_pct_error_sci

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results")
plot_dir = os.path.join(RESULTS_DIR, "prior_fwd_samples")
os.makedirs(plot_dir, exist_ok=True)

FLUX_OUTER = 0.1
DATA_PREFIX = "ReactionDiffusion"
N_SAMPLES = 4
SEED = 123  # distinct from the training-data seed (0)

row_labels = ["Input $m$", "True", "DeepONet", "DeepONet+Corr"]
row_label_colors = ["black", "#f68612", "#176fc1", "#9b1ec4"]


def mesh_triangles(domain):
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    return domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)


def main():
    mesh_data = build_mesh()
    domain = mesh_data.mesh
    Vm = functionspace(domain, ("Lagrange", 1))
    Vu = Vm
    elements = mesh_triangles(domain)
    nodes = domain.geometry.x[:, :2]

    prior_sampler = PriorSampler(Vm, 0.005, 0.2, seed=SEED)
    model = ReactionDiffusionModel(
        Vm, Vu, prior_sampler, logn_scale=0.25, logn_translate=0.0, flux_outer=FLUX_OUTER,
    )
    sm = load_surrogate_model("DeepONet", DATA_PREFIX, PROBLEM_DIR, model)

    np.random.seed(SEED)
    plt.ioff()
    w = model.empty_m()
    err_nop, err_corr = [], []

    for i in range(N_SAMPLES):
        w, _ = prior_sampler(w)
        m = model.transform_gaussian_pointwise(w)
        u_true = model.solveFwd(m=m, transform_m=False)
        u_nop = sm.predict_from_m(m)
        u_corr = model.residual_correct(m, u_nop)

        e_nop = np.linalg.norm(u_true - u_nop) / np.linalg.norm(u_true)
        e_corr = np.linalg.norm(u_true - u_corr) / np.linalg.norm(u_true)
        err_nop.append(e_nop)
        err_corr.append(e_corr)
        print(f"sample {i}: rel L2 error DeepONet={e_nop*100:.3f}%, DeepONet+correction={e_corr*100:.3f}%")

        quick_field_plot(m, nodes, elements=elements, cmap="jet",
                          savefilename=f"{plot_dir}/m_{i}.png", show_plot=False)
        plt.close()
        quick_field_plot(u_true, nodes, elements=elements, cmap="jet",
                          savefilename=f"{plot_dir}/u_true_{i}.png", show_plot=False)
        plt.close()
        quick_field_plot(u_nop, nodes, elements=elements, cmap="jet",
                          savefilename=f"{plot_dir}/u_nop_{i}.png", show_plot=False)
        plt.close()
        quick_field_plot(u_corr, nodes, elements=elements, cmap="jet",
                          savefilename=f"{plot_dir}/u_corr_{i}.png", show_plot=False)
        plt.close()

    idx = list(range(N_SAMPLES))
    row_paths = [
        [f"{plot_dir}/m_{i}.png" for i in idx],
        [f"{plot_dir}/u_true_{i}.png" for i in idx],
        [f"{plot_dir}/u_nop_{i}.png" for i in idx],
        [f"{plot_dir}/u_corr_{i}.png" for i in idx],
    ]
    blank = [None] * N_SAMPLES
    cell_titles = [
        blank,
        blank,
        [format_pct_error(err_nop[i]) for i in idx],
        [format_pct_error_sci(err_corr[i]) for i in idx],
    ]
    cell_title_colors = [
        blank,
        blank,
        [row_label_colors[2]] * N_SAMPLES,
        [row_label_colors[3]] * N_SAMPLES,
    ]

    montage_path = os.path.join(RESULTS_DIR, "reaction_diffusion_correction.png")
    save_image_montage(
        row_paths, montage_path,
        row_labels=row_labels, row_label_colors=row_label_colors,
        cell_titles=cell_titles, cell_title_colors=cell_title_colors,
        label_w=110, label_fontsize=30,
        title_fontsize=24, title_x_shift=0,
    )
    print(f"\nSaved to {montage_path}")


if __name__ == "__main__":
    main()
