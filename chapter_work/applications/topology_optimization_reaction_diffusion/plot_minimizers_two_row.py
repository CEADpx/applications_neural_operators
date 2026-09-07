"""
Two-row minimizer comparison, replotted from the saved three_way_comparison.npz:
    Row 1: the optimized m field from each forward map (true FE, DeepONet-only,
           DeepONet+correction). Columns 2/3 show the design error
               eps_a = ||m* - m*_a|| / ||m*|| * 100   (eq:optimizationDesignError)
    Row 2: the u solution at that minimizer, computed with the forward map
           that produced it -- column 2 is the raw DeepONet prediction at
           its own minimizer, not a true-FE re-evaluation. Columns 2/3 show
           the state error against the fixed reference u* = F(m*) (the
           true-FE state at the true-FE minimizer):
               e_a = ||u* - u*_a|| / ||u*|| * 100     (eq:optimizationStateError)

Montage style matches run_prior_forward_map_comparison.py: cmap='jet',
error titles via montage_util.py (\\varepsilon for design error, e for state
error).

Run in the 'neuralopv2' conda environment:
    python plot_minimizers_two_row.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, "..", "..", ".."))
PROBLEM_DIR = os.path.join(ROOT, "chapter_work", "problems", "reaction_diffusion")

sys.path.insert(0, PROBLEM_DIR)
sys.path.insert(0, NOTEBOOK_DIR)

from neural_operators.plotting.field_plot import quick_field_plot
from mesh_setup import build_mesh
from montage_util import save_image_montage, format_pct_error

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results")
plot_dir = os.path.join(RESULTS_DIR, "minimizer_cells")
os.makedirs(plot_dir, exist_ok=True)

variants = ["true_fe", "nop_only", "nop_corrected"]
col_labels = ["True FE", "DeepONet only", "DeepONet+Corr"]
col_colors = ["#f68612", "#176fc1", "#9b1ec4"]


def main():
    mesh_data = build_mesh()
    domain = mesh_data.mesh
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    elements = domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    nodes = domain.geometry.x[:, :2]

    data = np.load(os.path.join(RESULTS_DIR, "three_way_comparison.npz"))

    m_star = data["true_fe_m"]
    u_star = data["true_fe_u_true"]  # = F(m*), fixed reference for the state error

    plt.ioff()
    eps_pct = [None]
    e_pct = [None]
    for name in variants[1:]:
        m_a = data[f"{name}_m"]
        u_a = data[f"{name}_u_opt"]  # surrogate's own prediction at m*_a
        eps_pct.append(np.linalg.norm(m_star - m_a) / np.linalg.norm(m_star))
        e_pct.append(np.linalg.norm(u_star - u_a) / np.linalg.norm(u_star))
        print(f"{name}: eps_a={100*eps_pct[-1]:.2f}%  e_a={100*e_pct[-1]:.2f}%")

    for name in variants:
        m = data[f"{name}_m"]
        u_opt = data[f"{name}_u_opt"]
        quick_field_plot(m, nodes, elements=elements, cmap="jet",
                          savefilename=f"{plot_dir}/m_{name}.png", show_plot=False)
        plt.close()
        quick_field_plot(u_opt, nodes, elements=elements, cmap="jet",
                          savefilename=f"{plot_dir}/u_{name}.png", show_plot=False)
        plt.close()

    row_paths = [
        [f"{plot_dir}/m_{name}.png" for name in variants],
        [f"{plot_dir}/u_{name}.png" for name in variants],
    ]
    cell_titles = [
        [col_labels[0], f"{col_labels[1]}\n" + format_pct_error(eps_pct[1], symbol=r"\varepsilon"),
         f"{col_labels[2]}\n" + format_pct_error(eps_pct[2], symbol=r"\varepsilon")],
        [None, format_pct_error(e_pct[1]), format_pct_error(e_pct[2])],
    ]
    cell_title_colors = [
        col_colors,
        [None, col_colors[1], col_colors[2]],
    ]

    montage_path = os.path.join(RESULTS_DIR, "three_way_minimizers_two_row.png")
    save_image_montage(
        row_paths, montage_path,
        row_labels=["minimizer $m$", "solution $u$"],
        row_label_colors=["black", "black"],
        cell_titles=cell_titles, cell_title_colors=cell_title_colors,
        label_w=90, label_fontsize=32,
    )
    print(f"Saved to {montage_path}")


if __name__ == "__main__":
    main()
