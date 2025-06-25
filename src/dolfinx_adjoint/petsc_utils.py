import typing

from petsc4py import PETSc

import dolfinx


def solve_linear_problem(
    A: PETSc.Mat,  # type: ignore [name-defined]
    x: dolfinx.la.Vector,
    b: dolfinx.la.Vector,
    petsc_options: typing.Optional[dict] = None,
):
    """Solve a linear problem :math:`Ax = b`.

    Args:
        A: The matrix
        x: The solution vector
        b: The right-hand side vector
        petsc_options: Optional dictionary of PETSc options for the solver.
    """

    petsc_options = {} if petsc_options is None else petsc_options
    error_if_not_converged = petsc_options.pop("ksp_error_if_not_converged", True)
    petsc_options["ksp_error_if_not_converged"] = error_if_not_converged
    ksp = PETSc.KSP().create(A.comm)  # type: ignore [attr-defined]

    ksp.setOperators(A)

    # Give PETSc solver options a unique prefix
    problem_prefix = f"dolfinx_adjoint_linear_problem_{id(ksp)}"
    ksp.setOptionsPrefix(problem_prefix)

    # Set PETSc options
    opts = PETSc.Options()  # type: ignore [attr-defined]
    opts.prefixPush(problem_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    # Set matrix and vector PETSc options
    A.setOptionsPrefix(problem_prefix)
    A.setFromOptions()
    b.petsc_vec.setOptionsPrefix(problem_prefix)
    b.petsc_vec.setFromOptions()

    # Free option space post setting
    for k in petsc_options.keys():
        del opts[k]
    ksp.solve(b.petsc_vec, x.petsc_vec)
    ksp.destroy()
