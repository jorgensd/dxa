from mpi4py import MPI

import dolfinx.fem.petsc
import numpy as np
import ufl

apply_derivatives = ufl.algorithms.apply_derivatives.apply_derivatives


N = 5
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
uh = dolfinx.fem.Function(V)
du = ufl.TrialFunction(V)
dv = ufl.TestFunction(V)
m = dolfinx.fem.Function(Q)

m.interpolate(lambda x: x[0] + x[1])

F = ufl.inner(ufl.grad(uh), ufl.grad(dv)) * ufl.dx - ufl.inner(m, dv) * ufl.dx

alpha = dolfinx.fem.Constant(mesh, 1.0e-6)
x, y = ufl.SpatialCoordinate(mesh)
d = 1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)
J = 1 / 2 * ufl.inner(uh - d, uh - d) * ufl.dx + alpha / 2 * m**2 * ufl.dx

dFdu = ufl.derivative(F, uh, du)
dFdudm = ufl.derivative(dFdu, m)
d2Fdudu = ufl.derivative(dFdu, uh)
dFdm = ufl.derivative(F, m)
d2Fdmdm = ufl.derivative(dFdm, m)

dJdm = ufl.derivative(J, m)
dJdu = ufl.derivative(J, uh)
d2Jdmdu = ufl.derivative(dJdu, m)
d2Jdudu = ufl.derivative(dJdu, uh)
d2Jdmdm = ufl.derivative(dJdm, m)

u_dot = dolfinx.fem.Function(V)  # TLM solution
dm = dolfinx.fem.Function(Q)  # Perturbation direction
dm.interpolate(lambda x: np.sin(x[1]))
lmbda = dolfinx.fem.Function(V)  # Adjoint solution
lmbda_dot = dolfinx.fem.Function(V)  # Second order adjoint solution

# Solve forward problem
a, L = ufl.system(ufl.replace(F, {uh: du}))
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, exterior_facets)
bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), exterior_dofs, V)
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
forward_problem = dolfinx.fem.petsc.LinearProblem(a, L, u=uh, petsc_options=petsc_options, bcs=[bc])
forward_problem.solve()

# Solve TLM
a_tlm = dFdu
L_tlm = -ufl.action(dFdm, dm)
problem_tlm = dolfinx.fem.petsc.LinearProblem(a_tlm, L_tlm, u=u_dot, petsc_options=petsc_options, bcs=[bc])
problem_tlm.solve()

# Solve adjoint problem
a_adj = ufl.adjoint(dFdu)
L_adj = dJdu
problem_adj = dolfinx.fem.petsc.LinearProblem(a_adj, L_adj, u=lmbda, petsc_options=petsc_options, bcs=[bc])
problem_adj.solve()

# Solve second order adjoint problem
a_soa = ufl.adjoint(dFdu)
L_soa = ufl.action(ufl.adjoint(d2Jdudu), u_dot)  # + ufl.action(ufl.adjoint(d2Jdmdu), dm)
# - ufl.action(ufl.action(ufl.adjoint(d2Fdudu), u_dot), lmbda) - ufl.action(ufl.adjoint(ufl.action(d2Fdmdm,dm)),lmbda)
problem_soa = dolfinx.fem.petsc.LinearProblem(a_soa, L_soa, u=lmbda_dot, petsc_options=petsc_options, bcs=[bc])
problem_soa.solve()

# [delta m]^T  d^2 J / dm^2 [delta m]
Hmdm = -ufl.action(ufl.adjoint(dFdm), lmbda_dot) + ufl.action(
    ufl.adjoint(d2Jdmdm), dm
)  #  - ufl.action(ufl.action(ufl.adjoint(d2Fdmdm), dm))

# dJ/dm [delta m]
Jac_adj = -ufl.action(ufl.adjoint(dFdm), lmbda) + dJdm

Jac_vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(Jac_adj))
Hm_vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(Hmdm))


J_compiled = dolfinx.fem.form(J)

m_org = m.x.array.copy()
J_org = dolfinx.fem.assemble_scalar(dolfinx.fem.form(J_compiled))
step_length = 0.01
steps = [step_length * (1 / 2) ** i for i in range(4)]


dJac_dm = np.dot(Jac_vec.array, dm.x.array)
Hm_dm = np.dot(Hm_vec.array, dm.x.array)

errors = []
errors_der = []
errors_hess = []
for step in steps:
    m.x.array[:] = m_org + step * dm.x.array[:]

    forward_problem.solve()
    J_perturbed = dolfinx.fem.assemble_scalar(J_compiled)

    errors.append(J_perturbed - J_org)
    errors_der.append(J_perturbed - J_org - step * dJac_dm)
    errors_hess.append(J_perturbed - J_org - step * dJac_dm - step**2 / 2 * Hm_dm)
errors = np.abs(np.array(errors))
errors_der = np.abs(np.array(errors_der))
errors_hess = np.abs(np.array(errors_hess))


def convergence_rates(r, p):
    cr = []  # convergence rates
    for i in range(1, len(p)):
        cr.append(np.log(r[i] / r[i - 1]) / np.log(p[i] / p[i - 1]))
    return cr


print(errors, convergence_rates(errors, steps))
print(errors_der, convergence_rates(errors_der, steps))
print(errors_hess, convergence_rates(errors_hess, steps))
breakpoint()
