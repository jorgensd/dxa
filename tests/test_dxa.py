import dolfinx
import pyadjoint
import numpy
from dolfinx_adjoint import Function, assign, assemble_scalar

from mpi4py import MPI
import ufl
import pytest

@pytest.fixture(scope="module")
def mesh1():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

@pytest.fixture(scope="module")
def mesh2():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

@pytest.fixture(scope="module")
def mesh3():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

@pytest.mark.parametrize("mesh", [mesh1, mesh2, mesh3])
def test_assign(mesh):
    pyadjoint.set_working_tape(pyadjoint.Tape())
    pyadjoint.continue_annotation()




    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.name = "u_output"

    # Control variable
    d = pyadjoint.AdjFloat(0.2)

    v = Function(V)
    v.name = "v"
    assign(d, v)
    # FIXME: Add time dependent PDE


    assign(v, u)

    c = 0.3
    error = ufl.inner(u-c, u-c)*ufl.dx
    J = assemble_scalar(error)

    control = pyadjoint.Control(d)
    Jh = pyadjoint.ReducedFunctional(J, control)

    # DEBUG: Look at tape
    tape = pyadjoint.get_working_tape()
    tape.visualise_dot("testx.dot")

    # DEBUG check differentiation
    Jh.derivative()

    # DEBUG: Check the value of the functional
    for x in [0.2, 0.4, -0.2, 0.5, -1.3]:
        assert numpy.isclose(Jh(x), (x-c)**2)

    # DEBUG: Check minimzation call
    opt = pyadjoint.minimize(Jh, options={"maxiter": 10, "disp": True})
    print(Jh(opt))
    assert numpy.isclose(opt, c)