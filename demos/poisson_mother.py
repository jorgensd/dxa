# # Optimal Control of the Poisson equation
# *Section author: Jørgen S. Dokken [(dokken@simula.no)](mailto:dokken@simula.no)*.
#
# Original implementation in [dolfin-adjoint](https://github.com/dolfin-adjoint/dolfin-adjoint) was by Simon W. Funke.

# This demo solves the *mother problem* of PDE-constrained optimization: the optimal control of the Possion equation.
# Physically, this problem can be interpreted as finding the best heating/cooling of a cooktop to achieve a desired temperature profile.

# This example introduces the basics of how to solve optimization problems with DOLFINx-adjoint.

# ## Problem definition
# Mathematically, the goal is to minimize the following tracking type functional:
#
# $$
# \min_{f \in Q} J(u) = \frac{1}{2} \int_{\Omega} (u - d)^2 ~\mathrm{d}x + \frac{\alpha}{2}\int_{\Omega} f^2~\mathrm{d} x
# $$
# 
# subject to the Poisson equation with Dirichlet boundary conditions:
#
# $$
# \begin{align}
# - \kappa \Delta u &= f  && \text{in } \Omega, \\
# u &= 0 && \text{on } \partial\Omega, \\
# a &\leq f \leq b &&
# \end{align}
# $$
#
# where $\Omega$ is the domain of interest, $u: \Omega \mapsto \mathbb{R}$ is the unknown temperature,
# $\kappa\in\mathbb{R}$ is the thermal diffusivity, $f: \Omega \mapsto \mathbb{R}$ is the unknown contrl function acting as a source term,
# $d:\Omega\mapsto \mathbb{R}$ is the desired temperature profile, and $\alpha\in[0,\infty)$ is a Tikhonov regularization parameter,
# and $a,b\in\mathbb{R}$ are the lower and upper bounds on the control function $f$. 
# Note that $f(x)>0$ corresponds to heating, while $f(x)<0$ corresponds to cooling.

# It can be shown that this problem is well-posed and has a unique solution, see for instance Section 1.5 {cite}`Ulbrich2009` or {cite}`troltzsch2010optimal`.

# ## Implementation
# We start by import the necessary modules for this demo, which includes `mpi4py`, `dolfinx` and `dolfinx_adjoint`.

from mpi4py import MPI
import pyadjoint
import dolfinx
import dolfinx_adjoint

# Next we import [Moola](https://github.com/funsim/moola/), which is a Python package
# containing a collection of optimization solvers specifically designed for PDE-constrained optimization problems.

import moola

# Next we create a regular mesh of a unit square, which will be our domain $\Omega$.
# Some optimization algorithms suffer from bad performance when the mesh is non-uniform (i.e. when the mesh is partially refined).
# To demonstrate that Moola does not have this issue, we will refine our mesh in the center of the domain.
# We use the DOLFINx refine function to do this.

# +
n = 64
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)

def refinement_region(x, tol=1e-14):
    """Define a conditional to determine the refinement region.
    
    Args:
        x: Coordinates of mesh nodes, shape (num_nodes, 3)
        tol: Tolerance for being within the region.
    """
    clause_x = abs(x[0]-0.5) < 0.25 + tol
    clause_y = abs(x[1]-0.5) < 0.25 + tol
    return clause_x & clause_y

mesh.topology.create_connectivity(1, mesh.topology.dim)
edges_to_refine = dolfinx.mesh.locate_entities(
    mesh, 1, refinement_region
)
refined_mesh, _, _ = dolfinx.mesh.refine(mesh, edges_to_refine)
tdim = refined_mesh.topology.dim
del mesh
# -

# Next we use Pyvista to plot the mesh

# +
import pyvista
pyvista.set_jupyter_backend("html")
import sys, os
if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)

grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(refined_mesh))
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightgrey")
plotter.view_xy()
plotter.show()
# -

# Then we define the discrete function spaces $V$ and $Q$ for the state and control variable, respectively

V = dolfinx.fem.functionspace(refined_mesh, ("Lagrange", 1))
Q = dolfinx.fem.functionspace(refined_mesh, ("Discontinuous Lagrange", 0))

# The optimization algorithm will use the value of the control function $f$ as an initial guess for the optimization.
# A zero intial guess seems to be too simple: For example the L-BFGS algorithm will find the optimial control with just two iterations.
# To make it more interesting, we choose a non-zero initial guess.

f = dolfinx_adjoint.Function(Q, name="Control")
f.interpolate(lambda x: x[0] + x[1])  # Set intial guess

# ```{note}
# As opposed to standard DOLFINx code, we use `dolfinx_adjoint.Function` to create the control function.
# This is so that we can track it throughout the program on the computational tape.
# ```
# We also create a state variable that we will store the solution to the Poisson equation in.

uh = dolfinx_adjoint.Function(V, name="State")

# Next, we define the variational formulation of the Poisson equation.

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V) 
kappa = dolfinx.fem.Constant(refined_mesh, 1.0)  # Thermal diffusivity
F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
a, L = ufl.system(F)

# We create the Dirichlet BC

refined_mesh.topology.create_connectivity(tdim-1, tdim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(refined_mesh.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(V, tdim-1, exterior_facets)
bc = dolfinx.fem.dirichletbc(0.0, exterior_dofs, V)

# Next, we define a `dolfinx_adjoint.LinearProblem` instance, which overloads
# the `dolfinx.fem.petsc.LinearProblem` class.

petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
"ksp_error_if_not_converged": True}
problem = dolfinx_adjoint.LinearProblem(a, L, u=uh, bcs=[bc], petsc_options=petsc_options,
                                        adjoint_petsc_options=petsc_options)
problem.solve()

# ```{note}
# Note that we can pass in solver options for the adjoint equation via the keyword argument
# `adjoint_petsc_options`. As we a solving a linear, symmetric problem, i,e, a self-adjoint problem,
# we use the same options for both the forward and adjoint problems.
# ```

# Before we can start the optimization, we need to specity the control variable and the functional of interest.
# For this example we use $d(x,y)=\frac{1}{2\pi^2} \sin(\pi x)\sin( \pi y)$ as the desired temperature profile

x, y = ufl.SpatialCoordinate(refined_mesh)
d =  1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)

# The functional is written out in `ufl` and assembled with `dolfinx_adjoint.assemble_scalar`

alpha = dolfinx.fem.Constant(refined_mesh, 1.0e-6)  # Tikhonov regularization parameter
J_symbolic = 0.5*ufl.inner(uh-d, uh-d) * ufl.dx + 0.5*alpha*ufl.inner(f, f) * ufl.dx
J = dolfinx_adjoint.assemble_scalar(J_symbolic)

# The next step is to formulate the so-called reduced optimization problem.
# The idea is that the solution $u$ cna be considered as a functin of $f$: 
# given a value of $f$, we can solve the Poisson equation and obtain the associated solution $uh$.
# Bu denoting this solution function as $u(f)$ we can rewrite the original optimization problem as a reduced problem:

# $$
# \min_{f \in Q} \hat J(u(f), f) = \frac{1}{2} \int_{\Omega} (u(f) - d)^2 ~\mathrm{d}x + \frac{\alpha}{2}\int_{\Omega} f^2~\mathrm{d} x
# $$

# Note that no PDE-constraint is required anymore, since it is implicitly contained in the solution function.

# *dolfinx-adjoint* can automatically reduce the optimization problem by using `pyadjoint.ReducedFunctional`.
# This object solves the forward PDE using the `pyadjoint.Tape` each time the functional is evaluated.
# Additionally, it derives and solves the adjoint equation each time the function gradient should be evaluated.

control = pyadjoint.Control(f)
Jhat = pyadjoint.ReducedFunctional(J, control)
Jhat.derivative()

tol = 1e-6
opt = pyadjoint.minimize(
    Jhat,
    method="BFGS",
    tol=tol,
    scale=1,
    options={"maxiter": 200, "disp": True},
    derivative_options={
        "riesz_representation": "l2",
        "petsc_options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        "jit_options": {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]},
    },
)

# # Now that all ingredients are in place, we can perform the optimization.
# # For this, we employ the `moola.MoolaOptimizationProblem` to generate a problem that is compatible with the Moola framework.

# from dolfinx_adjoint.moola_adapters import DolfinxPrimalVector
# problem = pyadjoint.MoolaOptimizationProblem(rf)
# f_moola = DolfinxPrimalVector(f)

# # Then, we wrap the control function into a Moola object, and create a `moola.NewtonCG`` solver for solving the optimisation problem
# # FIXME: Add DOLFINx.la.Vector as block variable in DOLFINx-adjoint
# solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
#                                                    'maxiter': 20,
#                                                    'display': 3,
#                                                    'ncg_hesstol': 0})

# sol = solver.solve()
# f_opt = sol['control'].data
# print(f_opt)

# ## References
# ```{bibliography}
# :filter: cited and ({"demos/poisson_mother"} >= docnames)
# ```

