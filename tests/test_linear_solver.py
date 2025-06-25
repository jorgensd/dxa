import typing

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)


@pytest.mark.parametrize("constant", [np.float64(0.2)])  # , float(-0.13), int(3)])
@pytest.mark.parametrize("mesh_var_name", ["mesh_1D"])  # , "mesh_2D", "mesh_3D"])
def test_solver(mesh_var_name: str, request, constant: typing.Union[float, int, np.floating]):
    pyadjoint.set_working_tape(pyadjoint.Tape())
    pyadjoint.continue_annotation()

    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uh = Function(V, name="u_output")

    f = Function(V, name="control")
    k = Function(V, name="kappa")
    k.x.array[:] = 1.0
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=mesh)
    L = ufl.inner(f, v) * ufl.dx(domain=mesh)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(1.0, boundary_dofs, V)

    from dolfinx_adjoint.solvers import LinearProblem

    problem = LinearProblem(
        a,
        L,
        u=uh,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True},

    )
    problem.solve()

    d = pyadjoint.AdjFloat(constant)
    error = ufl.inner(uh - d, uh - d) * ufl.dx
    J = assemble_scalar(error)

    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    tape = pyadjoint.get_working_tape()
    d = Function(V)
    d.interpolate(lambda x: x[0])
    print(d.x.array)
    print(Jh(d))

    e = Function(V)
    e.interpolate(lambda x: np.sin(x[0]))
    print(e.x.array)
    print(Jh(e))
    return
    tape.visualise_dot("test_solver.dot")
    # assert np.isclose(Jh(d), (float(d) - c) ** 4)

    # # Check derivative
    # dJ = Jh.derivative()
    # assert np.isclose(dJ, 4 * (float(d) - c) ** 3)

    # # Perform taylor test
    # du = pyadjoint.AdjFloat(0.1)

    # # Without gradient
    # Jh(d)
    # min_rate = pyadjoint.taylor_test(Jh, d, du, dJdm=0)
    # assert numpy.isclose(min_rate, 1.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 1.0, got {min_rate}"

    # # With gradient
    # Jh(d)
    # min_rate = pyadjoint.taylor_test(Jh, d, du)
    # assert numpy.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    # Jh(d)
    # tol = 1e-9
    # opt = pyadjoint.minimize(
    #     Jh,
    #     method="BFGS",
    #     tol=tol,
    #     scale=1e10,
    #     options={"maxiter": 200, "disp": True},
    #     derivative_options={
    #         "riesz_representation": "l2",
    #     },
    # )
    # np.testing.assert_allclose(float(opt), float(c), atol=1e-5)
