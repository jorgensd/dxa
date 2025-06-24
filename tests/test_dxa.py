from IPython import embed
import dolfinx
import pyadjoint

from dolfinx_adjoint import Function, assign, assemble_scalar

from mpi4py import MPI
import ufl

pyadjoint.set_working_tape(pyadjoint.Tape())
pyadjoint.continue_annotation()

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = Function(V)
u.name = "u_output"

d = pyadjoint.AdjFloat(0.2)
assign(d, u)

c = 0.3
error = ufl.inner(u-c, u-c)*ufl.dx
J = assemble_scalar(error)

control = pyadjoint.Control(d)
Jh = pyadjoint.ReducedFunctional(J, control)

#Jh.derivative()

opt = pyadjoint.minimize(Jh)

breakpoint()

tape = pyadjoint.get_working_tape()
tape.visualise_dot("testx.dot")
embed()
