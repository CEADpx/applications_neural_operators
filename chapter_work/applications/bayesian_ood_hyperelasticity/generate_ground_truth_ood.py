"""
Generate an out-of-distribution ground truth for the hyperelasticity Bayesian
inversion problem (sec:bayesian, "Inference using a shifted prior").

survey_work/applications/bayesian_inverse_problem_hyperelasticity draws its
ground-truth w from the same prior used to train the neural operators, close
enough to the training distribution that all three surrogates already track
the reference posterior well. Here the true w is pushed further out (larger
multi-mode amplitude + prior-noise scale, see TRUTH_AMPLITUDE_SCALE /
TRUTH_PRIOR_NOISE_SCALE), while the inference prior (INFERENCE_PRIOR_AC /
INFERENCE_PRIOR_CC) can independently match or differ from the training
prior. The physical transform m(w) is kept identical for training, truth,
and inference (alpha_m=100, beta_m=1000) so the only effect under study is
neural-operator state error for a given m -- not a mismatched prior/transform.

Set USE_OOD_TRANSFORM_FOR_TRUTH = True to also shift alpha_m/beta_m for the
truth (sec:accuracy "OOD Case 2"); in that regime the FE reference chain is
itself biased relative to m_true, so it isn't comparable to the primary
result on the same footing. Off by default.

Run in the 'neuralopv2' conda environment from this directory:
    python generate_ground_truth_ood.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import numpy as np
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx.mesh import CellType
from mpi4py import MPI
from scipy.interpolate import griddata

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..', '..', '..'))
ELASTICITY_DIR = os.path.join(ROOT, "survey_work/problems/hyperelasticity")

sys.path.insert(0, ELASTICITY_DIR)

from neural_operators.prior.priorSampler import PriorSampler
from hyperelasticityModel import HyperelasticityModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 0

# training / inference-time prior (must match what the surrogates were
# trained on, so that DeepONet/PCANet/FNO remain applicable)
TRAIN_PRIOR_AC = 0.005
TRAIN_PRIOR_CC = 0.2

# training / reference transform: m(w) = LOGN_SCALE * exp(w) + LOGN_TRANSLATE
#
# Shifted from the surrogates' training transform (100,1000) to (100,1200),
# under the article's Case 1 shift (100,1000)->(100,1500). Calibrated
# forward-only, no MCMC: at this shift, DeepONet/PCANet/FNO raw state error
# given m averages 5.8-6.3% (FE converges 15/15), one residual-correction
# step cuts it to 0.6-1.7%. Must match ground truth, training data, and
# inference, or the FE reference chain's identifiability breaks; only the
# frozen surrogates (trained under (100,1000)) are OOD.
LOGN_SCALE = 100.0
LOGN_TRANSLATE = 1200.0

# --- knobs that create the distribution shift for the TRUE field ---
# amplitude multiplier on the deterministic multi-mode bump field used by
# survey_work's Generate_GroundTruth.ipynb (vin, vout there are 0.3, -0.3)
#
# (2.0, 1.5) overflows: the prior draw scaled by 1.5 occasionally pushes w
# large enough that m = 100*exp(w)+1000 overflows to ~1e21 at some nodes and
# the FE reference solve fails to converge. Swept (amp, noise) in
# {1.0..2.0} x {0.5..1.5}; (1.5, 1.0) is the mildest choice that keeps m
# bounded (m_true in [~1090, ~1270] vs. ~[1055,1190] for an in-distribution
# draw) and converges robustly (checked at 20/40/80 load steps and
# rtol 1e-8/1e-10, |u_true| agrees to within ~6%).
TRUTH_AMPLITUDE_SCALE = 1.5
# multiplier on the prior-draw contribution added on top of the bump field
# (0.5 in the in-distribution notebook)
TRUTH_PRIOR_NOISE_SCALE = 1.0

# set True to additionally generate truth under a different (alpha,beta)
# transform than training (sec:accuracy OOD Case 2: (1000, 1000)); see
# module docstring for why this changes the interpretation of the result
USE_OOD_TRANSFORM_FOR_TRUTH = False
OOD_LOGN_SCALE = 1000.0
OOD_LOGN_TRANSLATE = 1000.0

NX, NY = 50, 50
FE_ORDER = 1

RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "Results", "ground_truth")
os.makedirs(RESULTS_DIR, exist_ok=True)


def multi_mode_w_field(x, amplitude_scale=1.0):
    vin, vout = 0.3 * amplitude_scale, -0.3 * amplitude_scale
    a, b = vin, np.log(vin - vout)
    tol = 1.0e-12
    xc = [[0.7, 0.7, 0.24], [0.3, 0.3, 0.2]]
    val = 0.0
    for i in range(len(xc)):
        xx, yy = x[0] - xc[i][0], x[1] - xc[i][1]
        r = xc[i][2]
        f = (r * r - xx * xx - yy * yy) / (r * r)
        if f < 1.0 + tol:
            val = val + a * np.exp(-b * f)
        else:
            val = val + vout
    return val


def main():
    np.random.seed(SEED)

    domain = mesh.create_unit_square(MPI.COMM_WORLD, NX, NY, cell_type=CellType.triangle)
    Vm = functionspace(domain, ("Lagrange", FE_ORDER))
    Vu = functionspace(domain, ("Lagrange", FE_ORDER, (2,)))

    # prior_sampler here plays the role of the TRAINING prior; it is what
    # HyperelasticityModel needs for bookkeeping (e.g. computing its mean),
    # and it is reused below to add a (scaled) prior-noise contribution to
    # the deterministic bump field, exactly as in the in-distribution notebook.
    prior_sampler = PriorSampler(Vm, TRAIN_PRIOR_AC, TRAIN_PRIOR_CC, SEED)
    model = HyperelasticityModel(Vm, Vu, prior_sampler, LOGN_SCALE, LOGN_TRANSLATE, SEED)
    nodes = model.m_nodes
    print(f"m_dim={model.m_dim}, u_dim={model.u_dim}, nodes.shape={nodes.shape}")

    w_bump = np.array([multi_mode_w_field(x, TRUTH_AMPLITUDE_SCALE) for x in nodes])
    w_true = w_bump + TRUTH_PRIOR_NOISE_SCALE * prior_sampler()[0]

    np.save(os.path.join(RESULTS_DIR, "w_true.npy"), w_true)

    if USE_OOD_TRANSFORM_FOR_TRUTH:
        m_true = OOD_LOGN_SCALE * np.exp(w_true) + OOD_LOGN_TRANSLATE
    else:
        m_true = model.transform_gaussian_pointwise(w_true)

    u_true = model.solveFwd(u=None, m=m_true, transform_m=False)

    num_grid_x, num_grid_y = 16, 16
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.05, 0.95, num_grid_x),
        np.linspace(0.05, 0.95, num_grid_y),
        indexing="ij",
    )
    x_obs = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    num_nodes = nodes.shape[0]

    grid_w = griddata(nodes, w_true, (grid_x, grid_y), method="linear")
    grid_m = griddata(nodes, m_true, (grid_x, grid_y), method="linear")
    u_xy = u_true.reshape(num_nodes, 2)
    grid_u = np.zeros((num_grid_x, num_grid_y, 2))
    grid_u[:, :, 0] = griddata(nodes, u_xy[:, 0], (grid_x, grid_y), method="linear")
    grid_u[:, :, 1] = griddata(nodes, u_xy[:, 1], (grid_x, grid_y), method="linear")

    w_obs = grid_w.flatten()
    m_obs = grid_m.flatten()
    u_obs = grid_u.reshape(-1)

    np.savez(
        os.path.join(RESULTS_DIR, "data.npz"),
        num_grid_x=num_grid_x,
        num_grid_y=num_grid_y,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_w=grid_w,
        grid_m=grid_m,
        grid_u=grid_u,
        w_true=w_true,
        m_true=m_true,
        u_true=u_true,
        x_obs=x_obs,
        w_obs=w_obs,
        m_obs=m_obs,
        u_obs=u_obs,
        # bookkeeping for reproducibility / for writing up the setup
        truth_amplitude_scale=TRUTH_AMPLITUDE_SCALE,
        truth_prior_noise_scale=TRUTH_PRIOR_NOISE_SCALE,
        used_ood_transform_for_truth=USE_OOD_TRANSFORM_FOR_TRUTH,
        logn_scale=LOGN_SCALE if not USE_OOD_TRANSFORM_FOR_TRUTH else OOD_LOGN_SCALE,
        logn_translate=LOGN_TRANSLATE if not USE_OOD_TRANSFORM_FOR_TRUTH else OOD_LOGN_TRANSLATE,
    )

    print(f"Saved ground truth to {RESULTS_DIR}")
    print(f"|w_true| = {np.linalg.norm(w_true):.3e}, |m_true| = {np.linalg.norm(m_true):.3e}, "
          f"m_true range = [{m_true.min():.1f}, {m_true.max():.1f}]")
    print(f"|u_true| = {np.linalg.norm(u_true):.3e}")


if __name__ == "__main__":
    main()
