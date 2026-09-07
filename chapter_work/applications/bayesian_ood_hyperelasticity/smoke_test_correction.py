"""
Cheap sanity check for HyperelasticityModel.residual_correct and
CorrectedSurrogateModel -- run this BEFORE any MCMC chain.

No MCMC, no ground-truth files needed: draws a handful of w samples (some
in-distribution, some scaled up to mimic the OOD ground truth), and for each
one prints the relative L2 state error of the raw neural-operator prediction
vs. the residual-corrected prediction against the finite-element solution.

Expected outcome, consistent with sec:accuracy of the book chapter draft
("Residual correction of nonlinear predictions") and the topology-optimization
result: the corrected error should be one to two orders of magnitude smaller
than the raw error, and this gap should widen (not shrink) as the sample is
pushed further out of distribution. If corrected error is NOT smaller than
raw error, or is comparable to it, something in residual_correct's Newton-step
plumbing (sign convention, BC handling, traction reset) is wrong and should be
debugged before trusting any MCMC result built on top of it.

Run in the 'neuralopv2' conda environment:
    python smoke_test_correction.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import numpy as np
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx.mesh import CellType
from mpi4py import MPI

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, "survey_work/problems/hyperelasticity")

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
from priorSampler import PriorSampler
from correctedSurrogateModel import CorrectedSurrogateModel
from nn_util import load_surrogate_model

SEED = 0
PRIOR_AC, PRIOR_CC = 0.005, 0.2
LOGN_SCALE, LOGN_TRANSLATE = 100.0, 1000.0
NX, NY = 50, 50

# multiplies the raw prior draw before the log-normal transform, to push
# samples away from the training distribution (1.0 == in-distribution)
OOD_SCALES = [1.0, 2.0, 4.0]


def main():
    np.random.seed(SEED)

    domain = mesh.create_unit_square(MPI.COMM_WORLD, NX, NY, cell_type=CellType.triangle)
    Vm = functionspace(domain, ("Lagrange", 1))
    Vu = functionspace(domain, ("Lagrange", 1, (2,)))

    prior_sampler = PriorSampler(Vm, PRIOR_AC, PRIOR_CC, SEED)
    model = HyperelasticityModel(Vm, Vu, prior_sampler, LOGN_SCALE, LOGN_TRANSLATE, SEED)

    surrogate_models = {}
    for name, kwargs in [("DeepONet", {}), ("PCANet", {}), ("FNO", {"u_comps": 2})]:
        sm = load_surrogate_model(name, "Hyperelasticity", MODEL_DIR, model, **kwargs)
        if sm is not None:
            surrogate_models[name] = sm

    if not surrogate_models:
        raise RuntimeError("No trained surrogates found under survey_work/problems/hyperelasticity.")

    for scale in OOD_SCALES:
        w, _ = prior_sampler()
        w = scale * w
        m = model.transform_gaussian_pointwise(w)
        u_fe = model.solveFwd(u=None, m=m, transform_m=False)

        print(f"\n--- OOD scale {scale} (|w|={np.linalg.norm(w):.2f}, "
              f"m range=[{m.min():.1f},{m.max():.1f}]) ---")
        for name, sm in surrogate_models.items():
            u_raw = sm.solveFwd(w)
            err_raw = 100 * np.linalg.norm(u_raw - u_fe) / np.linalg.norm(u_fe)

            corrected = CorrectedSurrogateModel(sm)
            u_corr = corrected.solveFwd(w)
            err_corr = 100 * np.linalg.norm(u_corr - u_fe) / np.linalg.norm(u_fe)

            flag = "OK" if err_corr < err_raw else "CHECK ME"
            print(f"  {name:10s}: raw err = {err_raw:8.3f}%   corrected err = {err_corr:8.3f}%   [{flag}]")


if __name__ == "__main__":
    main()
