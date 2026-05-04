import typing

from mpi4py import MPI

import dolfinx
import numpy
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar, assign


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 50, 50, 50)


@pytest.mark.parametrize("constant", [np.float64(0.2), float(-0.13), int(3)])
@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
def test_assign_constant(mesh_var_name: str, request, constant: typing.Union[float, int, np.floating]):
    pyadjoint.get_working_tape().clear_tape()
    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    u = Function(V, name="u_output")

    # Control variable
    d = pyadjoint.AdjFloat(constant)

    assign(d, u)

    c = 0.3
    error = ufl.inner(u - c, u - c) * ufl.inner(u - c, u - c) * ufl.dx(domain=mesh)

    J = assemble_scalar(error)

    control = pyadjoint.Control(d)
    Jh = pyadjoint.ReducedFunctional(J, control)

    assert np.isclose(Jh(d), (float(d) - c) ** 4)

    # Check derivative
    dJ = Jh.derivative()
    assert np.isclose(dJ, 4 * (float(d) - c) ** 3)

    # Perform taylor test
    du = pyadjoint.AdjFloat(0.1)

    # Without gradient
    Jh(d)
    min_rate = pyadjoint.taylor_test(Jh, d, du, dJdm=0)
    assert numpy.isclose(min_rate, 1.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 1.0, got {min_rate}"

    # With gradient
    Jh(d)
    min_rate = pyadjoint.taylor_test(Jh, d, du)
    assert numpy.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    Jh(d)
    tol = 1e-9
    opt = pyadjoint.minimize(
        Jh,
        method="BFGS",
        tol=tol,
        scale=1e10,
        options={"maxiter": 200, "disp": True},
    )
    np.testing.assert_allclose(float(opt), float(c), atol=1e-5)
