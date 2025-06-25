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
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)


@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
def test_function_control(mesh_var_name: str, request):
    pyadjoint.set_working_tape(pyadjoint.Tape())
    pyadjoint.continue_annotation()

    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.name = "u_output"

    v = Function(V)
    v.name = "v"
    v.x.array[:] = 0.2

    x = ufl.SpatialCoordinate(mesh)
    c = x[0]
    error = ufl.inner(v - c, v - c) * ufl.dx(domain=mesh)

    J = assemble_scalar(error)

    control = pyadjoint.Control(v)
    Jh = pyadjoint.ReducedFunctional(J, control)
    assert Jh(u) > 0
    tol = 1e-9
    opt = pyadjoint.minimize(
        Jh,
        method="Newton-CG",
        tol=tol,
        scale=100,
        options={"maxiter": 200, "disp": True},
        derivative_options={
            "riesz_representation": "l2",
            "petsc_options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
            "jit_options": {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]},
        },
    )

    def u_ex(x):
        return x[0]

    u_opt = dolfinx.fem.Function(V)
    u_opt.interpolate(u_ex)
    assert numpy.isclose(Jh(opt), 0.0)
    numpy.testing.assert_allclose(opt.x.array, u_opt.x.array, rtol=1e-5, atol=1e-5)
