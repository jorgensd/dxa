from mpi4py import MPI

import dolfinx
import numpy
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)


@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
def test_assign(mesh_var_name: str, request):
    pyadjoint.set_working_tape(pyadjoint.Tape())
    pyadjoint.continue_annotation()

    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.name = "u_output"

    # Control variable
    # d = pyadjoint.AdjFloat(0.2)

    v = Function(V)
    v.name = "v"
    v.x.array[:] = 0.2
    # assign(d, v)
    # # FIXME: Add time dependent PDE

    # assign(v, u)
    x = ufl.SpatialCoordinate(mesh)
    c = x[0]  # 0.3
    p = 100
    error = p * ufl.inner(v - c, v - c) * ufl.dx(domain=mesh)
    # error = (u-c)*ufl.dx
    J = assemble_scalar(error)

    control = pyadjoint.Control(v)
    Jh = pyadjoint.ReducedFunctional(J, control)
    tape = pyadjoint.get_working_tape()
    tape.visualise_dot("test2.dot")
    # DEBUG: Look at tape

    # DEBUG: check differentiation
    dJdm_adj = Jh.derivative()
    print(dJdm_adj.x.array)
    # print(dJdm_adj, p * (float(d) - c))
    # breakpoint()
    # assert numpy.isclose(dJdm_adj, 2*(d-c))
    # breakpoint()

    # DEBUG: check tlm
    # NOTE: Need to overload `dolfinx.fem.Constant` to support Hessian/TLM of such a problem.
    h = dolfinx.fem.Function(V)
    h.interpolate(lambda x: numpy.sin(x[0]))
    Hval = Jh.hessian(h)
    print(Hval.x.array)
    # DEBUG: Check the value of the functional
    for x in [0.2, 0.4, -0.2, 0.5, -1.3]:
        x_vec = Function(V)
        x_vec.x.array[:] = x
        # assert numpy.isclose(Jh(x_vec), p*(x - c) ** 2)

    # DEBUG: Check minimzation call
    tol = 1e-9
    opt = pyadjoint.minimize(Jh, method="CG", tol=tol, options={"maxiter": 200, "disp": True})
    print(Jh(opt))
    print(opt.x.array)

    def u_ex(x):
        return x[0]

    u_opt = dolfinx.fem.Function(V)
    u_opt.interpolate(u_ex)

    numpy.testing.assert_allclose(opt.x.array, u_opt.x.array, rtol=100 * tol, atol=100 * tol)
    # breakpoint()
    # assert numpy.isclose(opt, c)
