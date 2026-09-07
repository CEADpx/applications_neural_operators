import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, log
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

from neural_operators.pde.pdeModel import PDEModel

from mesh_setup import is_inner_boundary


class ReactionDiffusionModel(PDEModel):
    """
    Nonlinear reaction-diffusion topology-optimization forward model,
    matching Jha CMAME (sec:topOpt) exactly:

        -div(m grad u) + u^3 = 0   in Omega,
        u = 0                       on Gamma_in  (both void boundaries, Dirichlet),
        m grad(u).n = g             on Gamma_out (outer square, flux, g=0.1).

    Compliance objective (single term, no volumetric source):
        J(m) = int_{Gamma_out} g * u(m) ds.

    m prior: m = 0.25*exp(w), w a Gaussian random field (logn_scale=0.25,
    logn_translate=0.0), matching CMAME's stated m parameterization.

    Dirichlet at both voids removes the pure-Neumann null space, so Newton
    starts safely from u=0.
    """

    def __init__(
        self,
        Vm,
        Vu,
        prior_sampler,
        logn_scale=0.25,
        logn_translate=0.0,
        seed=0,
        flux_outer=0.1,
        n_load_steps=1,
        u_init=0.0,
    ):
        super().__init__(Vm, Vu, prior_sampler, seed)

        self.logn_scale = logn_scale
        self.logn_translate = logn_translate

        self.newton_rtol = 1e-9
        self.newton_atol = 1e-9
        self.newton_max_it = 50
        self.n_load_steps = n_load_steps
        self.u_init = u_init

        qd = {"quadrature_degree": 4}
        dx = ufl.Measure("dx", domain=self.mesh, metadata=qd)
        ds = ufl.Measure("ds", domain=self.mesh, metadata=qd)
        self._dx = dx
        self._ds = ds

        self._flux_default = flux_outer
        self._flux = fem.Constant(self.mesh, default_scalar_type(flux_outer))

        dofs = fem.locate_dofs_geometrical(Vu, is_inner_boundary)
        self.bc = [fem.dirichletbc(default_scalar_type(0.0), dofs, Vu)]

        self.m_mean = self.compute_mean(self.m_mean)
        self.m_fn = fem.Function(self.Vm)
        self.vertex_to_function(self.m_mean, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.u_fn = fem.Function(self.Vu)
        self.u_test = ufl.TestFunction(self.Vu)

        self._residual_form = (
            self.m_fn * ufl.inner(ufl.grad(self.u_fn), ufl.grad(self.u_test)) * dx
            + (self.u_fn ** 3) * self.u_test * dx
            - self._flux * self.u_test * ds
        )

        # compliance J(m) = int_{Gamma_out} g*u ds (single flux term -- no
        # volumetric source, matching CMAME's plain compliance definition)
        self._compliance_form = fem.form(self._flux * self.u_fn * ds)

        self._nonlinear_problem = None
        self._newton_solver = None

    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        """Matches PoissonModel's interface convention, so notebook code that
        calls model.is_point_on_dirichlet_boundary (e.g. get_dirichlet_bc)
        works unchanged for this model too."""
        return is_inner_boundary(x)

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
                f"Reaction-diffusion (CMAME) Newton solver did not converge ({n_iter} iterations)."
            )
        return n_iter

    def assemble(self, assemble_lhs=True, assemble_rhs=True):
        """No-op for API compatibility with linear models."""

    def transform_gaussian_pointwise(self, w, m_local=None):
        if m_local is None:
            self.m_transformed = self.logn_scale * np.exp(w) + self.logn_translate
            return self.m_transformed.copy()
        return self.logn_scale * np.exp(w) + self.logn_translate

    def compute_mean(self, m):
        return self.transform_gaussian_pointwise(self.prior_sampler.mean, m)

    def solveFwd(self, u=None, m=None, transform_m=False, reset_u=True):
        if m is None:
            m = self.samplePrior()

        if transform_m:
            self.m_transformed = self.transform_gaussian_pointwise(m, self.m_transformed)
        else:
            self.m_transformed = m

        self.vertex_to_function(self.m_transformed, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        if reset_u:
            self.u_fn.x.array[:] = self.u_init

        n_steps = max(1, self.n_load_steps)
        for step in range(1, n_steps + 1):
            self._flux.value = self._flux_default * (step / n_steps)
            self._run_newton()
        self._flux.value = self._flux_default
        self._update_ghosts(self.u_fn)

        return self.function_to_vertex(self.u_fn, u, is_m=False)

    def compliance(self):
        return float(fem.assemble_scalar(self._compliance_form))

    def residual_correct(self, m, u_tilde, return_diagnostics=False):
        """One-Newton-step residual correction: start from a NOP-predicted
        u_tilde and take a single Newton step of the true physics residual
        at the given m, pulling it back toward the true solution. One step
        instead of a full solve from a generic initial guess. Used to build
        the "NOP + correction" forward map for the three-way optimization
        comparison."""
        self.vertex_to_function(m, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.vertex_to_function(u_tilde, self.u_fn, is_m=False)
        self._update_ghosts(self.u_fn)

        self._setup_solver()
        self._flux.value = self._flux_default

        res_before = None
        if return_diagnostics:
            res_before = self._residual_norm()

        saved_max_it = self._newton_solver.max_it
        self._newton_solver.max_it = 1
        try:
            self._newton_solver.solve(self.u_fn)
        except RuntimeError:
            pass
        self._newton_solver.max_it = saved_max_it

        self._update_ghosts(self.u_fn)
        u_c = self.function_to_vertex(self.u_fn, is_m=False)

        if return_diagnostics:
            res_after = self._residual_norm()
            correction_norm = float(np.linalg.norm(u_c - u_tilde))
            return u_c, (res_before, correction_norm, res_after)

        return u_c

    def _residual_norm(self):
        from dolfinx.fem.petsc import assemble_vector

        b = assemble_vector(fem.form(self._residual_form))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return float(np.linalg.norm(b.array))

    def samplePrior(self, m=None, transform_m=False):
        if transform_m:
            w, _ = self.prior_sampler()
            self.m_transformed = self.transform_gaussian_pointwise(w, self.m_transformed)
        else:
            self.m_transformed = self.prior_sampler()[0]

        if m is None:
            return self.m_transformed.copy()
        m = self.m_transformed.copy()
        return m
