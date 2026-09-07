"""
Three-way comparison for the material-field optimization application:
    (A) true FE forward solve (baseline),
    (B) DeepONet surrogate only (F_NOP in place of F inside the optimizer),
    (C) DeepONet surrogate + one-Newton-step residual correction.

Same problem/BCs/objective/optimizer settings across all three; only the
forward map inside optimize_material_field's loop changes (via the
forward_solver argument).

Comparison metric is the true compliance of each variant's final design,
evaluated with a full FE solve regardless of which forward_solver drove the
optimization -- not the self-reported J during optimization, which for
(B)/(C) is only as accurate as the surrogate at each iterate.

Run in the 'neuralopv2' conda environment:
    python run_three_way_comparison.py
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

sys.path.insert(0, PROBLEM_DIR)
sys.path.insert(0, NOTEBOOK_DIR)

from neural_operators.prior.priorSampler import PriorSampler
from neural_operators.nn.nn_util import load_surrogate_model
from mesh_setup import build_mesh
from reactionDiffusionModel import ReactionDiffusionModel
from topology_optimization import optimize_material_field

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ETA = 0.4
M0 = 0.5 * ETA
V_LW = 0.001
FLUX_OUTER = 0.1
FILTER_RADIUS = 0.012
N_OUTER_MAX = 300
MOVE_LIMIT = 0.1

DATA_PREFIX = "ReactionDiffusion"


def triangulation_from_mesh(domain):
    pts = domain.geometry.x
    cells = domain.topology.connectivity(domain.topology.dim, 0)
    tris = np.array([cells.links(i) for i in range(cells.num_nodes)])
    return mtri.Triangulation(pts[:, 0], pts[:, 1], tris)


def true_compliance(model, m):
    """Evaluate compliance under a genuine full FE solve -- the fair
    metric, independent of which forward_solver produced m."""
    u = model.solveFwd(m=m, transform_m=False)
    return model.compliance(), u


def make_model(Vm, Vu, prior_sampler):
    return ReactionDiffusionModel(
        Vm, Vu, prior_sampler, logn_scale=0.25, logn_translate=0.0, flux_outer=FLUX_OUTER,
    )


def main():
    mesh_data = build_mesh()
    domain = mesh_data.mesh
    Vm = functionspace(domain, ("Lagrange", 1))
    Vu = Vm
    triang = triangulation_from_mesh(domain)

    prior_sampler = PriorSampler(Vm, 0.005, 0.2, seed=0)

    variants = {}

    print("=" * 60)
    print("(A) true FE")
    print("=" * 60)
    model_a = make_model(Vm, Vu, prior_sampler)

    # load the already-trained DeepONet surrogate via the SAME infrastructure
    # used for the Bayesian inference application (neural_operators.nn.nn_util,
    # neural_operators.mcmc.surrogateModel) -- reuses its existing model/data
    # loading and encode/predict/decode pipeline rather than reimplementing it.
    sm = load_surrogate_model("DeepONet", DATA_PREFIX, PROBLEM_DIR, model_a)

    v_a, m_a, u_a, hist_a, _ = optimize_material_field(
        model_a, m0=M0, eta=ETA, v_lw=V_LW, filter_radius=FILTER_RADIUS,
        n_outer_max=N_OUTER_MAX, move=MOVE_LIMIT, verbose=False,
    )
    J_true_a, u_true_a = true_compliance(model_a, m_a)
    variants["true_fe"] = dict(v=v_a, m=m_a, u_opt=u_a, hist=hist_a, J_true=J_true_a, u_true=u_true_a)
    print(f"final self-reported J={hist_a['J'][-1]:.5e}, true-evaluated J={J_true_a:.5e}, "
          f"iters={len(hist_a['J'])}")

    print("=" * 60)
    print("(B) DeepONet only")
    print("=" * 60)
    model_b = make_model(Vm, Vu, prior_sampler)

    def fwd_nop(model_, m_):
        return sm.predict_from_m(m_)

    v_b, m_b, u_b, hist_b, _ = optimize_material_field(
        model_b, m0=M0, eta=ETA, v_lw=V_LW, filter_radius=FILTER_RADIUS,
        n_outer_max=N_OUTER_MAX, move=MOVE_LIMIT, verbose=False, forward_solver=fwd_nop,
    )
    J_true_b, u_true_b = true_compliance(model_b, m_b)
    variants["nop_only"] = dict(v=v_b, m=m_b, u_opt=u_b, hist=hist_b, J_true=J_true_b, u_true=u_true_b)
    print(f"final self-reported J={hist_b['J'][-1]:.5e}, true-evaluated J={J_true_b:.5e}, "
          f"iters={len(hist_b['J'])}")

    print("=" * 60)
    print("(C) DeepONet + residual correction")
    print("=" * 60)
    model_c = make_model(Vm, Vu, prior_sampler)

    def fwd_corrected(model_, m_):
        u_tilde = sm.predict_from_m(m_)
        return model_.residual_correct(m_, u_tilde)

    v_c, m_c, u_c, hist_c, _ = optimize_material_field(
        model_c, m0=M0, eta=ETA, v_lw=V_LW, filter_radius=FILTER_RADIUS,
        n_outer_max=N_OUTER_MAX, move=MOVE_LIMIT, verbose=False, forward_solver=fwd_corrected,
    )
    J_true_c, u_true_c = true_compliance(model_c, m_c)
    variants["nop_corrected"] = dict(v=v_c, m=m_c, u_opt=u_c, hist=hist_c, J_true=J_true_c, u_true=u_true_c)
    print(f"final self-reported J={hist_c['J'][-1]:.5e}, true-evaluated J={J_true_c:.5e}, "
          f"iters={len(hist_c['J'])}")

    print("\n" + "=" * 60)
    print("SUMMARY (true-evaluated compliance, lower is better; true FE is ground truth)")
    print("=" * 60)
    for name in ["true_fe", "nop_only", "nop_corrected"]:
        d = variants[name]
        rel_diff = 100 * (d["J_true"] - variants["true_fe"]["J_true"]) / variants["true_fe"]["J_true"]
        u_field_err = np.linalg.norm(d["u_opt"] - d["u_true"]) / np.linalg.norm(d["u_true"])
        print(f"{name:15s}: J_true={d['J_true']:.5e}  (%diff from true-FE: {rel_diff:+.2f}%)  "
              f"||u_opt-u_true||/||u_true|| at final design = {u_field_err:.4f}")

    np.savez(
        os.path.join(RESULTS_DIR, "three_way_comparison.npz"),
        **{f"{k}_{field}": v for k, d in variants.items() for field, v in d.items() if field != "hist"},
        m0=M0, eta=ETA,
    )

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name, title in zip(
        axs, ["true_fe", "nop_only", "nop_corrected"],
        ["(A) true FE", "(B) DeepONet only", "(C) DeepONet + correction"],
    ):
        d = variants[name]
        sc = ax.tripcolor(triang, d["m"], shading="gouraud", cmap="viridis")
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nJ_true={d['J_true']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "three_way_m_comparison.png"), dpi=150, bbox_inches="tight")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name, title in zip(
        axs, ["true_fe", "nop_only", "nop_corrected"],
        ["(A) true FE", "(B) DeepONet only", "(C) DeepONet + correction"],
    ):
        d = variants[name]
        sc = ax.tripcolor(triang, d["u_true"], shading="gouraud", cmap="inferno")
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nu at final design (true FE solve)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "three_way_u_comparison.png"), dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, label in zip(
        ["true_fe", "nop_only", "nop_corrected"],
        ["true FE", "DeepONet only", "DeepONet + correction"],
    ):
        ax.plot(variants[name]["hist"]["J"], label=label)
    ax.axhline(variants["true_fe"]["J_true"], color="gray", ls="--", lw=1, label="true FE final (true-evaluated)")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("self-reported J during optimization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "three_way_convergence.png"), dpi=150, bbox_inches="tight")

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
