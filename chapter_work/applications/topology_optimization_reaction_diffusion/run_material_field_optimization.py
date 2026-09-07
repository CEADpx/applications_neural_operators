"""
Material-property-field optimization, not topology optimization: the domain
(unit square minus the two circular holes) is filled everywhere with a real
host material of diffusivity m0 = 0.5*eta > 0. Optimizes a bounded
perturbation field v in [v_lw, 1] with mean(v) = eta, where

    m(x) = m0 + H(v)(x),   H = Helmholtz filter

is the physical diffusivity fed to the PDE, so m >= m0 > 0 everywhere --
no near-zero "void" as in SIMP's ersatz-material relaxation.

Same domain/BCs/objective as run_topology_optimization.py (Dirichlet at both
voids, outer flux g=0.1, J = int_Gamma_out g*u ds); only the
parameterization of the design variable changes.

Run in the 'neuralopv2' conda environment:
    python run_material_field_optimization.py
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
from topology_optimization import optimize_material_field

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ETA = 0.4
M0 = 0.5 * ETA          # host material diffusivity; must be < eta (feasibility)
V_LW = 0.001            # numerical floor on v, same role as CMAME's m_lw
FLUX_OUTER = 0.1

FILTER_RADIUS = 0.012
# m_tol=0.005 re-solved the volume constraint to only 0.5% precision each
# outer iteration, producing a ~2% oscillation in J; tightening to 1e-6
# (optimize_material_field's default) removed it -- monotonic convergence,
# not a limit cycle.
N_OUTER_MAX = 300
MOVE_LIMIT = 0.1
CHECKPOINTS = [1, 2, 3, 5, 10, 20, 50, 100, 150, 300]


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
    print(f"m_dim={model.m_dim}, u_dim={model.u_dim}, m0={M0}")

    v_opt, m_opt, u_opt, hist, snapshots = optimize_material_field(
        model, m0=M0, eta=ETA, v_lw=V_LW, filter_radius=FILTER_RADIUS,
        n_outer_max=N_OUTER_MAX, move=MOVE_LIMIT, checkpoints=CHECKPOINTS,
    )
    print(f"\nfinal: J={hist['J'][-1]:.5e}, mean(v)={hist['v_bar'][-1]:.4f}, "
          f"v range=[{v_opt.min():.4f}, {v_opt.max():.4f}], "
          f"m range=[{m_opt.min():.4f}, {m_opt.max():.4f}], iters={len(hist['J'])}")

    wall_s = np.array(hist["wall_s"])
    print(f"timing: total={wall_s.sum():.1f}s over {len(wall_s)} outer iters, "
          f"mean={wall_s.mean():.3f}s/iter")

    np.savez(
        os.path.join(RESULTS_DIR, "material_field_optimum.npz"),
        v=v_opt, m=m_opt, u=u_opt, J=hist["J"], v_bar=hist["v_bar"], lam=hist["lam"],
        v_change=hist["v_change"], wall_s=hist["wall_s"],
        filter_radius=FILTER_RADIUS, eta=ETA, m0=M0, v_lw=V_LW,
        **{f"snapshot_{k}": v for k, v in snapshots.items()},
    )

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    sc0 = axs[0].tripcolor(triang, m_opt, shading="gouraud", cmap="viridis")
    plt.colorbar(sc0, ax=axs[0], fraction=0.046)
    axs[0].set_aspect("equal")
    axs[0].set_title(f"optimal diffusivity m = m0 + H(v)\n"
                      f"m0={M0}, mean(m)={m_opt.mean():.3f}, range=[{m_opt.min():.3f},{m_opt.max():.3f}]")

    sc1 = axs[1].tripcolor(triang, u_opt, shading="gouraud", cmap="inferno")
    plt.colorbar(sc1, ax=axs[1], fraction=0.046)
    axs[1].set_aspect("equal")
    axs[1].set_title(f"temperature u at optimum\nJ={hist['J'][-1]:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "material_field_optimum_fields.png"), dpi=150, bbox_inches="tight")

    ckpts_sorted = sorted(snapshots.keys())
    n_panels = len(ckpts_sorted)
    fig, axs = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3))
    for ax, k in zip(axs, ckpts_sorted):
        v_k = snapshots[k]
        sc = ax.tripcolor(triang, v_k, shading="gouraud", cmap="gray_r", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"iter={k}\nmean(v)={v_k.mean():.3f}", fontsize=9)
    plt.colorbar(sc, ax=axs, fraction=0.015)
    plt.savefig(os.path.join(RESULTS_DIR, "material_field_v_trajectory.png"), dpi=150, bbox_inches="tight")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(hist["J"])
    axs[0].set_xlabel("outer iteration")
    axs[0].set_ylabel("J (compliance)")
    axs[1].plot(hist["v_bar"])
    axs[1].axhline(ETA, color="gray", ls="--", lw=1)
    axs[1].set_xlabel("outer iteration")
    axs[1].set_ylabel("mean(v)")
    axs[2].plot(hist["lam"])
    axs[2].set_xlabel("outer iteration")
    axs[2].set_ylabel("lambda")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "material_field_convergence.png"), dpi=150, bbox_inches="tight")

    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
