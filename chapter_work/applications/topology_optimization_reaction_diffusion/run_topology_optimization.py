"""
Jha CMAME topology optimization, reproduced with the exact original domain,
BCs (Dirichlet u=0 on both voids, flux g=0.1 on the outer boundary), and
compliance objective J(m) = int_{Gamma_out} g*u ds -- the only change from
the paper is a Helmholtz PDE filter applied to the sensitivity before the
SIMP update, addressing the spiky/poorly-regularized minimizer noted in the
book chapter's figure (figures/topology_minimizers.png).

Run in the 'neuralopv2' conda environment:
    python run_topology_optimization.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from dolfinx.fem import functionspace

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, "..", "..", ".."))
PROBLEM_DIR = os.path.join(ROOT, "chapter_work", "problems", "reaction_diffusion")

sys.path.insert(0, os.path.join(ROOT, "src", "prior"))
sys.path.insert(0, os.path.join(ROOT, "src", "pde"))
sys.path.insert(0, PROBLEM_DIR)

from mesh_setup import build_mesh
from priorSampler import PriorSampler
from reactionDiffusionModel import ReactionDiffusionModel
from topology_optimization import optimize

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# CMAME's stated values
ETA = 0.4
M_LW = 0.001
FLUX_OUTER = 0.1

# calibrated this session (see topology_optimization.py docstring): reaches
# the true bounds [m_lw,1] with no checkerboard noise
FILTER_RADIUS = 0.012
N_OUTER_MAX = 150
MOVE_LIMIT = 0.1

# checkpoints to characterize the true-FE m trajectory (statistics + timing)
# before calibrating any NOP training prior on it
CHECKPOINTS = [1, 2, 3, 5, 10, 20, 50, 100, 150]


def triangulation_from_mesh(domain):
    pts = domain.geometry.x
    cells = domain.topology.connectivity(domain.topology.dim, 0)
    tris = np.array([cells.links(i) for i in range(cells.num_nodes)])
    return mtri.Triangulation(pts[:, 0], pts[:, 1], tris)


def main():
    mesh_data = build_mesh()
    domain = mesh_data.mesh
    fe_order = 1
    Vm = functionspace(domain, ("Lagrange", fe_order))
    Vu = functionspace(domain, ("Lagrange", fe_order))
    triang = triangulation_from_mesh(domain)

    prior_sampler = PriorSampler(Vm, 0.005, 0.2, seed=0)
    model = ReactionDiffusionModel(
        Vm, Vu, prior_sampler, logn_scale=0.25, logn_translate=0.0,
        flux_outer=FLUX_OUTER,
    )
    print(f"m_dim={model.m_dim}, u_dim={model.u_dim}")

    m_opt, u_opt, hist, snapshots = optimize(
        model, eta=ETA, m_lw=M_LW, filter_radius=FILTER_RADIUS,
        n_outer_max=N_OUTER_MAX, move=MOVE_LIMIT, checkpoints=CHECKPOINTS,
    )
    print(f"\nfinal: J={hist['J'][-1]:.5e}, mean(m)={hist['m_bar'][-1]:.4f}, "
          f"m range=[{m_opt.min():.4f}, {m_opt.max():.4f}], iters={len(hist['J'])}")

    wall_s = np.array(hist["wall_s"])
    print(f"timing: total={wall_s.sum():.1f}s over {len(wall_s)} outer iters, "
          f"mean={wall_s.mean():.3f}s/iter, min={wall_s.min():.3f}s, max={wall_s.max():.3f}s")

    print("\nm-trajectory statistics (checkpoint iter: mean, std, min, max):")
    for k in sorted(snapshots.keys()):
        mk = snapshots[k]
        print(f"  iter={k:4d}: mean={mk.mean():.4f}  std={mk.std():.4f}  "
              f"min={mk.min():.4f}  max={mk.max():.4f}")

    np.savez(
        os.path.join(RESULTS_DIR, "cmame_topology_optimum.npz"),
        m=m_opt, u=u_opt, J=hist["J"], m_bar=hist["m_bar"], lam=hist["lam"],
        m_change=hist["m_change"], wall_s=hist["wall_s"],
        filter_radius=FILTER_RADIUS, eta=ETA, m_lw=M_LW,
        **{f"snapshot_{k}": v for k, v in snapshots.items()},
    )

    ckpts_sorted = sorted(snapshots.keys())
    n_panels = len(ckpts_sorted)
    fig, axs = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3))
    for ax, k in zip(axs, ckpts_sorted):
        mk = snapshots[k]
        sc = ax.tripcolor(triang, mk, shading="gouraud", cmap="gray_r", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"iter={k}\nmean={mk.mean():.3f}", fontsize=9)
    plt.colorbar(sc, ax=axs, fraction=0.015)
    plt.savefig(os.path.join(RESULTS_DIR, "cmame_m_trajectory.png"), dpi=150, bbox_inches="tight")

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    sc0 = axs[0].tripcolor(triang, m_opt, shading="gouraud", cmap="gray_r", vmin=0, vmax=1)
    plt.colorbar(sc0, ax=axs[0], fraction=0.046)
    axs[0].set_aspect("equal")
    axs[0].set_title(f"optimal diffusivity m (CMAME BCs, Helmholtz-filtered)\n"
                      f"r_filt={FILTER_RADIUS}, mean={hist['m_bar'][-1]:.3f}")

    sc1 = axs[1].tripcolor(triang, u_opt, shading="gouraud", cmap="inferno")
    plt.colorbar(sc1, ax=axs[1], fraction=0.046)
    axs[1].set_aspect("equal")
    axs[1].set_title(f"temperature u at optimum\nJ={hist['J'][-1]:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmame_topology_optimum_fields.png"), dpi=150, bbox_inches="tight")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(hist["J"])
    axs[0].set_xlabel("outer iteration")
    axs[0].set_ylabel("J (compliance)")
    axs[1].plot(hist["m_bar"])
    axs[1].axhline(ETA, color="gray", ls="--", lw=1)
    axs[1].set_xlabel("outer iteration")
    axs[1].set_ylabel("mean(m)")
    axs[2].plot(hist["lam"])
    axs[2].set_xlabel("outer iteration")
    axs[2].set_ylabel("lambda")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmame_topology_optimum_convergence.png"), dpi=150, bbox_inches="tight")

    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
