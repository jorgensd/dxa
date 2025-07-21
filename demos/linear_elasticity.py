# %% [markdown]
# # Data assimilation in a linear elasticity problem
# *Section author: Henrik Finsberg ([henriknf@simula.no](mailto:henriknf@simula.no))*.
#
# In this example we will demonstrate how to find an unknown pressure and a
# material parameter given some measurements of the displacement field. This problem serve a good example of
# an inverse problem where you have some data and want to estimate some unknown model parameters.
#
# ## Problem definition
# Mathematically, the goal is to minimize the following cost function
#
# $$
# \min_{(p, \mu) \in \mathbb{R}^2} J(u) = \frac{1}{2} \int_{\Omega} (u(p, \mu) - u_{\mathrm{data}})^2 ~\mathrm{d}x
# $$
#
# subject to the equation the linear elasticity equation
#
# $$
# \begin{align}
# \mathrm{div}(\sigma(u)) &= 0  && \text{in } \Omega, \\
# u &= 0 && \text{on } \partial\Omega_D, \\
# \sigma(u) \cdot n &= -p n && \text{on } \partial\Omega_N, \\
# \end{align}
# $$
#
# where
#
# $$
# \begin{align}
# \sigma(u) &= 2\mu \varepsilon(u) + \lambda \mathrm{div}(u)I, \\
# \varepsilon(u) &= \frac{1}{2}\left( \nabla u + \nabla u^T \right).
# \end{align}
# $$
#
# Here $\Omega$ is the domain of interest, $u: \Omega \mapsto \mathbb{R}^3$ is the
# unknown displacement field, $p \in\mathbb{R}$ is a scalar pressure acting normal
# to the Neumann boundary $\partial\Omega_N$, $\mu \in \mathbb{R}$ and $\lambda \in
# \mathbb{R}$ are the Lamé parameters and $u_{\mathrm{data}}:\Omega\mapsto \mathbb{R}^3$
# is some measured displacement field. In this formulation we will treat the pair of $p$
# and $\mu$ as the control which we assume is unknown, while the parameter $\lambda$ is
# assumed known. The goal is therefore to find $p$ and $\mu$ which minismises the
# model-data mismatch $J(u)$.
#
#
# We will start by importing the neccessary libraries.
# In particular we will import import [Moola](https://github.com/funsim/moola/), which is a Python package
# containing a collection of optimization solvers specifically designed for
# PDE-constrained optimization problems and [scifem](https://github.com/scientificcomputing/scifem/) which contains
# a collection of tools for scientific computing with a focus on finite element methods and will be used to
# create a real function space for the control parameters.

# %%
import os
import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import gmsh
import moola
import numpy as np
import pyadjoint
import pyvista
import scifem
import ufl
from moola.adaptors import DolfinxPrimalVector

import dolfinx_adjoint

comm = MPI.COMM_WORLD


# %% [markdown]
# We will now create an annulus mesh using gmsh which
# will have an outer radius of 50 mm and in inner radius of 10 mm.
# The characteristic length will control the level of resolution of the mesh
# (a smaller characteristic length will result in a finer mesh).


# %%
def generate_mesh(
    path: Path = Path("annulus.msh"),
    c=(0.0, 0.0, 0.0),  # origin
    r_outer: float = 50,
    r_inner: float = 10,
    char_length=10.0,
    verbose=True,
):
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("annulus")

    outer_id = gmsh.model.occ.addSphere(*c, r_outer)
    inner_id = gmsh.model.occ.addSphere(*c, r_inner)

    outer = [(3, outer_id)]
    inner = [(3, inner_id)]

    annulus, _ = gmsh.model.occ.cut(outer, inner, removeTool=False)

    surfaces = gmsh.model.occ.getEntities(dim=2)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(
        dim=surfaces[0][0],
        tags=[surfaces[0][1]],
        tag=1,
        name="INNER",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[1][0],
        tags=[surfaces[1][1]],
        tag=2,
        name="OUTER",
    )

    gmsh.model.add_physical_group(dim=3, tags=[t[1] for t in annulus], tag=3, name="WALL")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()


c = (0.0, 0.0, 0.0)
r_outer: float = 50
r_inner: float = 10
char_length = 10.0
msh_file = Path("annulus.msh")

if not msh_file.exists():
    generate_mesh(
        msh_file,
        r_inner=r_inner,
        r_outer=r_outer,
        c=c,
        verbose=False,
        char_length=char_length,
    )
comm.barrier()

# %% [markdown]
# We can read in this mesh using the dolfinx API

# %%
msh = dolfinx.io.gmshio.read_from_msh(msh_file, comm=comm)

# %% [markdown]
# We configure Pyvista for rendering

# %%
pyvista.set_jupyter_backend("html")
if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)

# %% [markdown]
# and visualize the mesh using a clip plane

# %%
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh.mesh))
plotter = pyvista.Plotter()
plotter.add_mesh_clip_plane(grid, show_edges=True, crinkle=True, color="lightgrey")
plotter.view_yz(plotter)
if pyvista.OFF_SCREEN:
    plotter.screenshot("linear_elasticity_mesh.png")
else:
    plotter.show()

# %% [markdown]
# We can also visualize the facet tags which will be used when setting the boundary conditions.

# %%
assert msh.facet_tags is not None, "Mesh does not have facet tags"
bgrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh.mesh, msh.facet_tags.dim, msh.facet_tags.indices))
bgrid.cell_data["Facet tags"] = msh.facet_tags.values
bgrid.set_active_scalars("Facet tags")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, opacity=0.2)
plotter.add_mesh_clip_plane(bgrid, show_edges=True, crinkle=True)
plotter.view_yz(plotter)
if pyvista.OFF_SCREEN:
    plotter.screenshot("linear_elasticity_facet_tags.png")
else:
    plotter.show()

# %% [markdown]
# We can now define the volume and surface measures which is used for integration

# %%
quad_degree = 4
dx = ufl.dx(msh.mesh, metadata={"quadrature_degree": quad_degree})
ds = ufl.ds(domain=msh.mesh, subdomain_data=msh.facet_tags, metadata={"quadrature_degree": quad_degree})


# %% [markdown]
# ## Generation synthetic data
#
# We will now generate an artificial displacement field $u_{\mathrm{data}}$ with a known pressure
# $p = 300$ Pa and $\mu = 1000$ Pa (for a real world application this will typically come from
# some measurements). The goal of the data assimilation is then to retrieve those values.
# We will fix $\lambda = 10 000$ Pa

# %%
p = dolfinx.fem.Constant(msh.mesh, dolfinx.default_scalar_type(300.0))
mu = dolfinx.fem.Constant(msh.mesh, dolfinx.default_scalar_type(1000.0))
lmbda = dolfinx.fem.Constant(msh.mesh, dolfinx.default_scalar_type(10_000.0))

# %% [markdown]
# We will use second order Lagrange elements for the displacement field

# %%
d = msh.mesh.geometry.dim
V = dolfinx.fem.functionspace(msh.mesh, ("CG", 2, (d,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# %% [markdown]
# We define the variational forms and set the pressure on the inner boundary using the
# physical groups from gmsh

# %%
I = ufl.Identity(d)  # noqa: E741
eps = lambda u: 0.5 * (ufl.grad(u) + ufl.grad(u).T)  # noqa: E731
sigma = 2 * mu * eps(u) + lmbda * ufl.div(u) * I

# Define variational problem
a = ufl.inner(sigma, eps(v)) * dx
n = ufl.FacetNormal(msh.mesh)
L = ufl.inner(-p * n, v) * ds(msh.physical_groups["INNER"][1])

# %% [markdown]
# We will now set the Diriclet boundary condition on the outer boundary

# %%
print("Locating outer boundary dofs")
outer_dofs = dolfinx.fem.locate_dofs_topological(
    V, msh.facet_tags.dim, msh.facet_tags.find(msh.physical_groups["OUTER"][1])
)
print("Applying Dirichlet BC on the outer boundary")
bcs = [dolfinx.fem.dirichletbc(np.array((0.0,) * d), outer_dofs, V)]

# %% [markdown]
# Now we create a function `u_data` that will hold the values of the synthetically
# generated solution, pass this to the problem and solve

# %%
print("Creating linear problem")
u_data = dolfinx_adjoint.Function(V, name="Data")
problem = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    u=u_data,
    bcs=bcs,
    petsc_options_prefix="dxa_demo_linear_elasticity_",
    petsc_options={
        "ksp_type": "cg",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-8,
        "ksp_max_it": 10000,
        "pc_type": "gamg",
        "ksp_monitor": None,
    },
)
print("Solving linear problem")
problem.solve()


# %% [markdown]
# ## Inversion
#
# We will now solve the inverse problem and we start by using `scifem` to set
# up a real function space for the controls ($p$ and $\mu$), and we initialize
# these to 0 Pa and 500 Pa respectively (just to pick something that is not the correct solution).

# %%
V_control = scifem.create_real_functionspace(msh.mesh, value_shape=(2,))
v_control = dolfinx_adjoint.Function(V_control, name="Control")
v_control.x.array[0] = 0.0  # initial pressure set to zero
v_control.x.array[1] = 500.0  # initial shear modulus set to 500
p_control = v_control[0]
mu_control = v_control[1]


# %% [markdown]
# We will now create the variational for for the inverse problem using the control variables

# %%
sigma_opt = 2 * mu_control * eps(u) + lmbda * ufl.div(u) * I
a_opt = ufl.inner(sigma_opt, eps(v)) * dx
L_opt = ufl.inner(-p_control * n, v) * ds(msh.physical_groups["INNER"][1])

# %% [markdown]
# We create a variable for the computed displacement field


# %%
u = dolfinx_adjoint.Function(V, name="Displacement Field")

# %% [markdown]
# And create the problem

# %%
problem_opt = dolfinx_adjoint.LinearProblem(
    a_opt,
    L_opt,
    u=u,
    bcs=bcs,
    petsc_options={
        "ksp_type": "cg",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-8,
        "ksp_max_it": 10000,
        "pc_type": "gamg",
    },
)
print("Solving linear problem")
problem_opt.solve()

# %% [markdown]
# We can now define the cost function

# %%
J_symbolic = 0.5 * ufl.inner(u - u_data, u - u_data) * ufl.dx
J = dolfinx_adjoint.assemble_scalar(J_symbolic)


# %% [markdown]
# and create the control and reduced functional

# %%
control = pyadjoint.Control(v_control)
Jhat = pyadjoint.ReducedFunctional(J, control)


# %% [markdown]
# Finally we create an optimization problem

# %%
optimization_problem = pyadjoint.MoolaOptimizationProblem(Jhat)
f_moola = DolfinxPrimalVector(v_control)

# %% [markdown]
# and solve it using the L-BFGS method

# %%
optimization_options = {"jtol": 0, "gtol": 1e-9, "Hinit": "default", "maxiter": 100, "mem_lim": 10, "rjtol": 0}
solver = moola.BFGS(optimization_problem, f_moola, options=optimization_options)
solution = solver.solve()

# %% [markdown]
# We can now retrieve the optimal control values and see that they are close to the values that generated the
# synthetic displacement field

# %%
p_sol, mu_sol = solution["control"].array()
print(f"Optimized pressure: {p_sol}, Optimized shear modulus: {mu_sol}")
print(f"Relative difference (p) {abs(p_sol - p.value) / p.value}")
print(f"Relative difference (mu) {abs(mu_sol - mu.value) / mu.value}")
