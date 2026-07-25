import os
import sys

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, log
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_root, "src", "pde"))
from pdeModel import PDEModel


class HyperelasticityModel(PDEModel):
    """
    2D compressible hyperelasticity on the unit square (Neo-Hookean energy).

    Same setup as ``LinearElasticityModel``: uncertain scalar Young's modulus
    field ``m(x)``, left-edge Dirichlet clamp ``u = 0`` on ``Γ_{u_d}``, body force zero,
    and uniform traction on ``Γ_{u_q} = ∂D_u \\ Γ_{u_d}``. Strain energy is the stable
    compressible Neo-Hookean form ``(μ/2)(I₁ - 3 - 2 ln J) + (λ/2)(ln J)²``.
    Uses a Newton solve each forward evaluation. When ``reset_u`` is True,
    traction is ramped in ``n_load_steps`` increments for robust convergence.
    """

    def __init__(
        self,
        Vm,
        Vu,
        prior_sampler,
        logn_scale=1.0,
        logn_translate=0.0,
        seed=0
    ):
        super().__init__(Vm, Vu, prior_sampler, seed)

        self.logn_scale = logn_scale
        self.logn_translate = logn_translate
        self.nu = 0.45
        self.lam_fact = self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_fact = 1.0 / (2 * (1 + self.nu))

        # Newton solver parameters
        self.newton_rtol = 1e-8
        self.newton_atol = 1e-8
        self.newton_max_it = 50
        self.reset_u = True
        self.n_load_steps = 20

        qd = {"quadrature_degree": 4}
        dx = ufl.Measure("dx", domain=self.mesh, metadata=qd)
        ds = ufl.Measure("ds", domain=self.mesh, metadata=qd)

        # External body force and boundary traction
        self.b = ufl.as_vector((0.0, 0.0))
        self._traction_x = fem.Constant(self.mesh, default_scalar_type(20.0))
        self._traction_y = fem.Constant(self.mesh, default_scalar_type(50.0))
        self.t = ufl.as_vector((self._traction_x, self._traction_y))
        # target (fully-loaded) traction, restored before a residual correction
        self._traction_default = (float(self._traction_x.value), float(self._traction_y.value))

        dofs = fem.locate_dofs_geometrical(Vu, self._dirichlet_boundary)
        zero = np.zeros(2, dtype=default_scalar_type)
        self.bc = [fem.dirichletbc(zero, dofs, Vu)]

        self.m_mean = self.compute_mean(self.m_mean)

        self.m_fn = fem.Function(self.Vm)
        self.vertex_to_function(self.m_mean, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.u_fn = fem.Function(self.Vu)
        self.u_test = ufl.TestFunction(self.Vu)

        # variational form
        spatial_dim = self.mesh.geometry.dim
        I = ufl.variable(ufl.Identity(spatial_dim))
        F = ufl.variable(I + ufl.grad(self.u_fn))
        C = ufl.variable(F.T * F)
        Ic = ufl.variable(ufl.tr(C))
        J = ufl.variable(ufl.det(F))

        mu = self.m_fn * self.mu_fact
        lam = self.m_fn * self.lam_fact
        # Stable compressible Neo-Hookean
        W = (mu / 2.0) * (Ic - 3.0 - 2.0 * ufl.ln(J)) + (lam / 2.0) * (ufl.ln(J)) ** 2
        P = ufl.diff(W, F)

        self._residual_form = (
            ufl.inner(self.b, self.u_test) * dx
            + ufl.inner(self.t, self.u_test) * ds
            - ufl.inner(ufl.grad(self.u_test), P) * dx
        )

        self._nonlinear_problem = None
        self._newton_solver = None

    @staticmethod
    def _dirichlet_boundary(x):
        return np.isclose(x[0], 0.0, atol=1e-10)

    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        tol = 1e-10
        if (
            np.abs(x[0]) < tol
            or np.abs(x[1]) < tol
            or np.abs(x[0] - 1.0) < tol
            or np.abs(x[1] - 1.0) < tol
        ):
            if x[0] < tol:
                return True
        return False

    def _update_ghosts(self, fn):
        fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def _setup_solver(self):
        if self._nonlinear_problem is None:
            log.set_log_level(log.LogLevel.WARNING)
            self._nonlinear_problem = NewtonSolverNonlinearProblem(
                self._residual_form,
                self.u_fn,
                bcs=self.bc,
            )
            self._newton_solver = NewtonSolver(self.mesh.comm, self._nonlinear_problem)
            self._newton_solver.rtol = self.newton_rtol
            self._newton_solver.atol = self.newton_atol
            self._newton_solver.max_it = self.newton_max_it
            self._newton_solver.convergence_criterion = "incremental"

    def _run_newton(self):
        self._setup_solver()
        n_iter, converged = self._newton_solver.solve(self.u_fn)
        if not converged:
            raise RuntimeError(
                f"Hyperelasticity Newton solver did not converge ({n_iter} iterations)."
            )
        return n_iter

    def set_traction(self, tx, ty):
        self._traction_x.value = tx
        self._traction_y.value = ty

    def assemble(self, assemble_lhs=True, assemble_rhs=True):
        """No-op for API compatibility with linear models."""

    def transform_gaussian_pointwise(self, w, m_local=None):
        if m_local is None:
            self.m_transformed = self.logn_scale * np.exp(w) + self.logn_translate
            return self.m_transformed.copy()
        return self.logn_scale * np.exp(w) + self.logn_translate

    def compute_mean(self, m):
        return self.transform_gaussian_pointwise(self.prior_sampler.mean, m)

    def solveFwd(self, u=None, m=None, transform_m=False):
        if m is None:
            m = self.samplePrior()

        if transform_m:
            self.m_transformed = self.transform_gaussian_pointwise(
                m, self.m_transformed
            )
        else:
            self.m_transformed = m

        self.vertex_to_function(self.m_transformed, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        if (
            abs(float(self._traction_x.value)) < 1e-14
            and abs(float(self._traction_y.value)) < 1e-14
        ):
            self.u_fn.x.array[:] = 0.0
            self._update_ghosts(self.u_fn)
            return self.function_to_vertex(self.u_fn, u, is_m=False)

        if self.reset_u:
            self.u_fn.x.array[:] = 0.0

        target_tx = float(self._traction_x.value)
        target_ty = float(self._traction_y.value)
        n_steps = self.n_load_steps if self.reset_u else 1

        for step in range(1, n_steps + 1):
            if n_steps > 1:
                load_frac = step / n_steps
                self._traction_x.value = target_tx * load_frac
                self._traction_y.value = target_ty * load_frac
            self._run_newton()

        self._traction_x.value = target_tx
        self._traction_y.value = target_ty
        self._update_ghosts(self.u_fn)

        return self.function_to_vertex(self.u_fn, u, is_m=False)

    def residual_correct(self, m, u_tilde, return_diagnostics=False):
        """
        One Newton step of the hyperelastic residual, taken from a supplied
        predicted state ``u_tilde`` rather than from zero. This is the
        residual-based correction: solve
            delta_u R(m, u_tilde)(d^c) = -R(m, u_tilde)
        for d^c and return u^c = u_tilde + d^c. See sec:correction in the
        book chapter draft; this is exactly eq:corrected_state applied to
        the hyperelastic residual form defined in __init__.

        Reuses the module's own NewtonSolver (already exercised by every
        forward solve in this repository) with max_it forced to 1, instead
        of re-implementing PETSc residual/tangent assembly from scratch.
        This keeps correctness tied to code that is already validated by
        solveFwd, rather than to a new, unverified low-level assembly path.
        Same pattern (max_it=1, error_on_nonconvergence=False) used for the
        hyperelastic single-Newton-step corrector in the companion
        agent_neural_operator repository.

        Parameters
        ----------
        m : ndarray, vertex-ordered physical modulus field (already transformed,
            i.e. m = alpha_m*exp(w) + beta_m -- same convention as solveFwd(transform_m=False)).
        u_tilde : ndarray, vertex-ordered predicted displacement (e.g. neural-operator output).
        return_diagnostics : if True, also return (residual_norm_before, correction_norm,
            residual_norm_after) for the practical diagnostics described in sec:correction.

        Returns
        -------
        u_c : ndarray, vertex-ordered corrected displacement.
        """
        self.vertex_to_function(m, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.vertex_to_function(u_tilde, self.u_fn, is_m=False)
        self._update_ghosts(self.u_fn)

        self._setup_solver()

        # correction is defined at the fully-loaded state, cf. eq:topology_states
        self._traction_x.value, self._traction_y.value = self._traction_default

        res_before = None
        if return_diagnostics:
            res_before = self._residual_norm()

        saved_max_it = self._newton_solver.max_it
        saved_error_flag = getattr(self._newton_solver, "error_on_nonconvergence", None)
        self._newton_solver.max_it = 1
        if saved_error_flag is not None:
            self._newton_solver.error_on_nonconvergence = False
            num_its, _converged = self._newton_solver.solve(self.u_fn)
            if num_its != 1:
                print(f"residual_correct: Newton solver ran {num_its} iterations instead of 1")
        else:
            # older dolfinx without error_on_nonconvergence: solve() raises on
            # nonconvergence, but u_fn is updated in-place before that check.
            try:
                self._newton_solver.solve(self.u_fn)
            except RuntimeError:
                pass
        self._newton_solver.max_it = saved_max_it
        if saved_error_flag is not None:
            self._newton_solver.error_on_nonconvergence = saved_error_flag

        self._update_ghosts(self.u_fn)
        u_c = self.function_to_vertex(self.u_fn, is_m=False)

        if return_diagnostics:
            res_after = self._residual_norm()
            correction_norm = float(np.linalg.norm(u_c - u_tilde))
            return u_c, (res_before, correction_norm, res_after)

        return u_c

    def _residual_norm(self):
        """L2 norm of the assembled residual vector at the current (m_fn, u_fn)."""
        from dolfinx.fem.petsc import assemble_vector as _assemble_vector

        b = _assemble_vector(fem.form(self._residual_form))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return float(np.linalg.norm(b.array))

    def samplePrior(self, m=None, transform_m=False):
        if transform_m:
            w, _ = self.prior_sampler()
            self.m_transformed = self.transform_gaussian_pointwise(
                w, self.m_transformed
            )
        else:
            self.m_transformed = self.prior_sampler()[0]

        if m is None:
            return self.m_transformed.copy()
        m = self.m_transformed.copy()
        return m
