"""
Mesh for the Jha CMAME nonlinear reaction-diffusion topology-optimization
problem, matching the domain in the book chapter (sec:topOpt): unit square
minus two circular voids,
    Omega = (0,1)^2 - B(x_c1, R1) - B(x_c2, R2),
    x_c1=(0.2,0.8), R1=0.1;  x_c2=(0.7,0.3), R2=0.2.

Both void boundaries are Dirichlet (u=0), so no facet tagging is needed: the
outer boundary is the only non-Dirichlet (flux) piece, and an unmarked `ds`
measure integrates over it exclusively once the void dofs are removed by the
Dirichlet condition (same as survey_work/problems/poisson/poissonModel.py).

Run standalone to sanity-check mesh generation and save a plot.
"""
import os

import gmsh
import numpy as np
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI

X_C1, R1 = (0.2, 0.8), 0.1
X_C2, R2 = (0.7, 0.3), 0.2
MESH_SIZE = 0.02  # ~similar resolution order to CMAME's 20614-element mesh

DOMAIN_TAG = 1


def build_mesh(mesh_size=MESH_SIZE, comm=MPI.COMM_WORLD, gdim=2):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)

    if comm.rank == 0:
        gmsh.model.add("reaction_diffusion_topopt_cmame")
        square = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        disk1 = gmsh.model.occ.addDisk(X_C1[0], X_C1[1], 0, R1, R1)
        disk2 = gmsh.model.occ.addDisk(X_C2[0], X_C2[1], 0, R2, R2)
        domain, _ = gmsh.model.occ.cut([(2, square)], [(2, disk1), (2, disk2)])
        gmsh.model.occ.synchronize()

        for dim, tag in domain:
            gmsh.model.addPhysicalGroup(dim, [tag], tag=DOMAIN_TAG)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.25)
        gmsh.model.mesh.generate(2)

    mesh_data = dolfinx_gmsh.model_to_mesh(gmsh.model, comm, 0, gdim=gdim)
    gmsh.finalize()
    return mesh_data


def is_inner_boundary(x, tol=1e-3):
    """Geometric Dirichlet predicate: true on either void's boundary."""
    r1 = np.sqrt((x[0] - X_C1[0]) ** 2 + (x[1] - X_C1[1]) ** 2)
    r2 = np.sqrt((x[0] - X_C2[0]) ** 2 + (x[1] - X_C2[1]) ** 2)
    return np.isclose(r1, R1, atol=tol) | np.isclose(r2, R2, atol=tol)


def is_inside_any_void(x):
    """True for points strictly inside either void disk, not just near the
    boundary curve (unlike is_inner_boundary). Needed when interpolating FE
    data onto a regular grid for FNO: scipy.interpolate.griddata's linear
    interpolation is defined over the convex hull of the FE mesh nodes,
    which for mesh nodes surrounding a hole is the full square -- griddata
    silently bridges values straight across the void. Grid points where
    this predicate is True must be zero-filled (or masked) after griddata,
    for both m and u."""
    r1 = np.sqrt((x[0] - X_C1[0]) ** 2 + (x[1] - X_C1[1]) ** 2)
    r2 = np.sqrt((x[0] - X_C2[0]) ** 2 + (x[1] - X_C2[1]) ** 2)
    return (r1 < R1) | (r2 < R2)


if __name__ == "__main__":
    mesh_data = build_mesh()
    domain = mesh_data.mesh
    print(f"num_vertices={domain.geometry.x.shape[0]}, "
          f"num_cells={domain.topology.index_map(domain.topology.dim).size_local}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts = domain.geometry.x
    cells = domain.topology.connectivity(domain.topology.dim, 0)
    tris = np.array([cells.links(i) for i in range(cells.num_nodes)])
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(triang, lw=0.3, color="steelblue")
    ax.set_aspect("equal")
    ax.set_title(f"{cells.num_nodes} elements, {pts.shape[0]} vertices")
    outdir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(outdir, "mesh_preview.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {os.path.join(outdir, 'mesh_preview.png')}")
