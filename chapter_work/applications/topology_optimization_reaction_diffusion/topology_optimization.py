"""
SIMP-style topology optimization of the diffusivity field m in the nonlinear
reaction-diffusion problem, matching Jha CMAME (sec:topOpt / s:topOptDetails):
same domain, Dirichlet-at-voids/flux-outer BCs, compliance objective
J(m) = int_{Gamma_out} g*u(m) ds, volume constraint mean(m)=eta with
m in [m_lw, 1]:

    minimize   J(m)
    subject to u(m) solves the reaction-diffusion residual,
               mean(m) = eta,  m_lw <= m <= 1.

Bi-level scheme (outer: update u given m; inner: bisection on the Lagrange
multiplier lambda to satisfy the volume constraint) follows CMAME's
Algorithm 1/2, m_new = clip(sqrt(e/lambda), m_lw, 1) with e = (m*grad(u))^2.

Adds a Helmholtz PDE filter on the sensitivity e before this update (SIMP +
PDE filter, Lazarov & Sigmund 2011), which CMAME's plain pointwise update
lacks; addresses the spiky, poorly-regularized minimizer seen without it.
No Heaviside projection, no passive design zones.
"""
import time

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc


class HelmholtzFilter:
    """Solve (-r^2 lap(e_tilde) + e_tilde = e) with homogeneous Neumann BC,
    i.e. the standard PDE density/sensitivity filter with radius r."""

    def __init__(self, V, radius):
        self.V = V
        self.radius = radius
        mesh = V.mesh
        qd = {"quadrature_degree": 4}
        dx = ufl.Measure("dx", domain=mesh, metadata=qd)

        e_trial = ufl.TrialFunction(V)
        e_test = ufl.TestFunction(V)
        r2 = fem.Constant(mesh, default_scalar_type(radius ** 2))
        self._a_form = fem.form(
            r2 * ufl.inner(ufl.grad(e_trial), ufl.grad(e_test)) * dx
            + ufl.inner(e_trial, e_test) * dx
        )
        self.e_fn = fem.Function(V)
        self._L_form = fem.form(ufl.inner(self.e_fn, e_test) * dx)

        self.lhs = assemble_matrix(self._a_form)
        self.lhs.assemble()
        self._ksp = PETSc.KSP().create(mesh.comm)
        self._ksp.setType(PETSc.KSP.Type.PREONLY)
        self._ksp.getPC().setType(PETSc.PC.Type.LU)
        self._ksp.setOperators(self.lhs)

        self.out_fn = fem.Function(V)

    def apply(self, e_vertex, vertex_to_function, function_to_vertex):
        vertex_to_function(e_vertex, self.e_fn, is_m=True)
        self.e_fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        rhs = assemble_vector(self._L_form)
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.out_fn.x.petsc_vec.set(0.0)
        self._ksp.solve(rhs, self.out_fn.x.petsc_vec)
        self.out_fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        return function_to_vertex(self.out_fn, is_m=True)


def compute_sensitivity(model, m_vertex, u_vertex):
    """e = (m * grad(u)) . (m * grad(u)), projected (L2) onto the Vm (P1) space.
    This is the pointwise quantity CMAME's update formula sqrt(e/lambda) uses."""
    mesh = model.mesh
    qd = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=qd)

    m_fn = model.vertex_to_function(m_vertex, fem.Function(model.Vm), is_m=True)
    u_fn = model.vertex_to_function(u_vertex, fem.Function(model.Vu), is_m=False)

    q = m_fn * ufl.grad(u_fn)
    e_expr = ufl.inner(q, q)

    trial = ufl.TrialFunction(model.Vm)
    test = ufl.TestFunction(model.Vm)
    a_form = fem.form(ufl.inner(trial, test) * dx)
    L_form = fem.form(ufl.inner(e_expr, test) * dx)

    A = assemble_matrix(a_form)
    A.assemble()
    b = assemble_vector(L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.setOperators(A)

    e_fn = fem.Function(model.Vm)
    ksp.solve(b, e_fn.x.petsc_vec)
    e_fn.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    return model.function_to_vertex(e_fn, is_m=True)


def m_update_from_sensitivity(e_vertex, lam, m_lw, m_up=1.0, m_prev=None, move=0.2):
    """SIMP optimality-criteria update, with an added move limit (standard in
    OC/SIMP schemes, e.g. Sigmund's 88-line code) capping how far m can change
    per iteration. Without it this bi-level scheme (as in CMAME, which does
    not mention a move limit) tends to oscillate between two nearby states
    rather than converging monotonically."""
    m_target = np.sqrt(np.maximum(e_vertex, 0.0) / lam)
    if m_prev is not None:
        lo = np.maximum(m_lw, m_prev - move)
        hi = np.minimum(m_up, m_prev + move)
        return np.clip(m_target, lo, hi)
    return np.clip(m_target, m_lw, m_up)


def bisection_inner(e_vertex, eta, m_lw, m_up, lam0, m_tol,
                     lam_max_iter=60, m_prev=None, move=0.2):
    """Bisection on lambda so that mean(m) equals eta, matching CMAME's
    Algorithm 2 (bracket then bisect)."""

    lam = lam0
    m = m_update_from_sensitivity(e_vertex, lam, m_lw, m_up, m_prev=m_prev, move=move)
    m_bar = float(np.mean(m))

    if m_bar < eta:
        lam_min = lam
        it = 0
        while m_bar < eta and it < lam_max_iter:
            lam = lam / 2.0
            m = m_update_from_sensitivity(e_vertex, lam, m_lw, m_up, m_prev=m_prev, move=move)
            m_bar = float(np.mean(m))
            it += 1
        lam_max = lam
    else:
        lam_max = lam
        it = 0
        while m_bar > eta and it < lam_max_iter:
            lam = lam * 2.0
            m = m_update_from_sensitivity(e_vertex, lam, m_lw, m_up, m_prev=m_prev, move=move)
            m_bar = float(np.mean(m))
            it += 1
        lam_min = lam

    it = 0
    while abs(m_bar - eta) > m_tol and it < lam_max_iter:
        lam_mid = 0.5 * (lam_min + lam_max)
        m = m_update_from_sensitivity(e_vertex, lam_mid, m_lw, m_up, m_prev=m_prev, move=move)
        m_bar = float(np.mean(m))
        if m_bar < eta:
            lam_min = lam_mid
        else:
            lam_max = lam_mid
        lam = lam_mid
        it += 1

    return m, lam, m_bar


def optimize(
    model,
    eta=0.4,
    m_lw=0.001,
    m_up=1.0,
    filter_radius=0.012,
    gamma_tol=1e-4,
    m_tol=0.005,
    n_outer_max=150,
    lam0=1.0,
    move=0.1,
    verbose=True,
    checkpoints=None,
):
    """CMAME's bi-level SIMP scheme (init m=0.1 uniform, per the paper) plus
    the Helmholtz sensitivity filter. filter_radius=0.012: r=0 reaches the
    true bounds [m_lw,1] but shows checkerboard/speckle noise; r=0.04
    over-smooths and never reaches the bounds; r in [0.008,0.016] all reach
    the bounds with no visible noise.

    checkpoints: optional list/set of outer-iteration counts (1-indexed) at
    which to save a copy of m -- used to characterize how the true-FE
    design trajectory evolves (statistics, timing) before calibrating any
    NOP training prior on it. Returns snapshots as {iter: m_array}.
    """
    m = np.full(model.m_dim, 0.1)
    lam = lam0
    history = {"J": [], "m_bar": [], "lam": [], "m_change": [], "wall_s": []}
    snapshots = {0: m.copy()}
    checkpoints = set(checkpoints) if checkpoints else set()

    hfilter = HelmholtzFilter(model.Vm, filter_radius) if filter_radius > 0 else None

    for k in range(n_outer_max):
        t0 = time.perf_counter()
        u = model.solveFwd(m=m, transform_m=False)
        J = model.compliance()

        e = compute_sensitivity(model, m, u)
        if hfilter is not None:
            e = hfilter.apply(e, model.vertex_to_function, model.function_to_vertex)

        m_new, lam, m_bar = bisection_inner(
            e, eta, m_lw, m_up, lam, m_tol=m_tol, m_prev=m, move=move,
        )
        wall_s = time.perf_counter() - t0

        m_change = float(np.linalg.norm(m_new - m) / max(1, len(m)) ** 0.5)
        history["J"].append(J)
        history["m_bar"].append(m_bar)
        history["lam"].append(lam)
        history["m_change"].append(m_change)
        history["wall_s"].append(wall_s)

        if verbose:
            print(f"[outer {k:3d}] J={J:.5e}  mean(m)={m_bar:.4f}  lambda={lam:.4e}  "
                  f"||m_new-m||_rms={m_change:.3e}  t={wall_s:.3f}s")

        m = m_new
        if (k + 1) in checkpoints:
            snapshots[k + 1] = m.copy()
        if m_change < gamma_tol:
            break

    u_final = model.solveFwd(m=m, transform_m=False)
    return m, u_final, history, snapshots


def optimize_material_field(
    model,
    m0,
    eta=0.4,
    v_lw=0.001,
    filter_radius=0.012,
    gamma_tol=1e-4,
    m_tol=1e-6,
    n_outer_max=300,
    lam0=1.0,
    move=0.1,
    damping=1.0,
    verbose=True,
    checkpoints=None,
    forward_solver=None,
):
    """
    Material-property-field optimization, not topology optimization. The
    domain (unit square minus the two circular holes) is filled everywhere
    with a real host material of diffusivity m0 > 0; nothing is ever void.
    Optimizes a bounded perturbation field v in [v_lw, 1] (v_lw a small
    positive floor -- the multiplicative OC update below is a fixed point at
    x=0 regardless of gradient, so v can't be allowed to reach exactly 0),
    subject to mean(v) = eta, with

        m(x) = m0 + H(v)(x),   H = Helmholtz filter (radius filter_radius).

    m0 must be < eta for mean(v)=eta to be reachable: H preserves the
    spatial mean exactly, so mean(m) = m0 + eta once v satisfies its
    constraint. m0 = 0.5*eta keeps this feasible.

    Since v >= v_lw > 0 and H preserves non-negativity, m >= m0 > 0
    everywhere -- no near-zero region, unlike SIMP's ersatz-void relaxation.

    Sensitivity: dJ/dm = -e_phys/m^2 (e_phys=(m grad u)^2, same quantity as
    optimize()'s e); dJ/dv = H(dJ/dm) by the filter's self-adjointness. The
    update on v reuses m_update_from_sensitivity/bisection_inner unchanged,
    with e replaced by e_v = (-dJ/dv)*v^2 = H(e_phys/m^2)*v^2.

    damping: under-relaxation in (0,1] applied after the move-limited
    OC/bisection update: v <- damping*v_new_raw + (1-damping)*v_old. Not
    needed to fix oscillation at m_tol=0.005/n_outer_max=150 -- that came
    from the inner bisection only resolving the volume constraint to m_tol
    precision each outer iteration; tightening m_tol to 1e-6 (the default)
    removed it (J decreases monotonically, no limit cycle). Damping alone
    only slows convergence further.

    forward_solver: optional callable (model, m_vertex) -> u_vertex, used in
    place of model.solveFwd for the forward map inside the optimization loop
    (e.g. a NOP surrogate, or a NOP surrogate + residual_correct). Defaults
    to the true FE solve. Its output is written into model.u_fn before
    compliance/sensitivity, so the three-way comparison (true FE / NOP-only
    / NOP+correction) swaps only the forward map.
    """
    if forward_solver is None:
        def forward_solver(model_, m_):
            return model_.solveFwd(m=m_, transform_m=False)

    v = np.full(model.m_dim, eta)
    lam = lam0
    history = {"J": [], "v_bar": [], "lam": [], "v_change": [], "wall_s": []}
    snapshots = {0: v.copy()}
    checkpoints = set(checkpoints) if checkpoints else set()

    hfilter = HelmholtzFilter(model.Vm, filter_radius)

    def physical_m(v_):
        Hv = hfilter.apply(v_, model.vertex_to_function, model.function_to_vertex)
        return m0 + Hv

    for k in range(n_outer_max):
        t0 = time.perf_counter()
        m = physical_m(v)
        u = forward_solver(model, m)
        model.vertex_to_function(u, model.u_fn, is_m=False)
        model._update_ghosts(model.u_fn)
        J = model.compliance()

        e_phys = compute_sensitivity(model, m, u)  # (m grad u)^2
        neg_dJdm = e_phys / m ** 2                  # = |grad u|^2, m>0 always here
        neg_dJdv = hfilter.apply(neg_dJdm, model.vertex_to_function, model.function_to_vertex)
        e_v = neg_dJdv * v ** 2

        v_new_raw, lam, v_bar_raw = bisection_inner(
            e_v, eta, v_lw, 1.0, lam, m_tol=m_tol, m_prev=v, move=move,
        )
        v_new = damping * v_new_raw + (1.0 - damping) * v if damping < 1.0 else v_new_raw
        v_bar = float(np.mean(v_new))
        wall_s = time.perf_counter() - t0

        v_change = float(np.linalg.norm(v_new - v) / max(1, len(v)) ** 0.5)
        history["J"].append(J)
        history["v_bar"].append(v_bar)
        history["lam"].append(lam)
        history["v_change"].append(v_change)
        history["wall_s"].append(wall_s)

        if verbose:
            print(f"[outer {k:3d}] J={J:.5e}  mean(v)={v_bar:.4f}  lambda={lam:.4e}  "
                  f"||v_new-v||_rms={v_change:.3e}  t={wall_s:.3f}s")

        v = v_new
        if (k + 1) in checkpoints:
            snapshots[k + 1] = v.copy()
        if v_change < gamma_tol:
            break

    m_final = physical_m(v)
    u_final = forward_solver(model, m_final)
    return v, m_final, u_final, history, snapshots
