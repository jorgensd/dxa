try:
    import typing_extensions as typing
except ModuleNotFoundError:
    import typing  # type: ignore[no-redef]

import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl

from dolfinx_adjoint.types import Function

from .blocks.solvers import LinearProblemBlock


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.

    Args:
        a: The bilinear form representing the left-hand side of the equation.
        L: The linear form representing the right-hand side of the equation.
        bcs: Boundary conditions to apply to the problem.
        u: Solution vector.
        P: Preconditioner for the linear problem.
        kind: Kind of PETSc Matrix to assemble the system into.
        petsc_options: Options dictionary for the PETSc krylov supspace solver.
        form_compiler_options: Form compiler options for generating assembly kernels.
        jit_options: Options for just-in-time compilation of the forms.
        entity_maps: Mapping from meshes that coefficients and arguments are defined on to the
            integration domain of the forms.
        ad_block_tag: Tag for adjoint blocks in the tape.
        adjoint_petsc_options: PETSc options for adjoint problems.
        tlm_petsc_options: Optional PETSc options for TLM problems.
    """

    def __init__(
        self,
        a: typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]],
        L: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
        bcs: typing.Optional[typing.Iterable[dolfinx.fem.DirichletBC]] = None,
        u: typing.Optional[typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]]] = None,
        P: typing.Optional[typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]]] = None,
        kind: typing.Optional[typing.Union[str, typing.Iterable[typing.Iterable[str]]]] = None,
        petsc_options: typing.Optional[dict] = None,
        petsc_options_prefix: str = "dxa_linear_problem_",
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        entity_maps: typing.Optional[dict[dolfinx.mesh.Mesh, npt.NDArray[np.int32]]] = None,
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
    ) -> None:
        self.ad_block_tag = ad_block_tag
        self._adj_options = adjoint_petsc_options
        self._tlm_options = tlm_petsc_options
        if u is None:
            try:
                # Extract function space for unknown from the right hand
                # side of the equation.
                assert isinstance(L, ufl.Form)
                self._u = Function(L.arguments()[0].ufl_function_space())
            except AttributeError:
                assert isinstance(L, typing.Iterable)
                self._u = [Function(Li.arguments()[0].ufl_function_space()) for Li in L]  # type: ignore[assignment]
        else:
            if isinstance(u, dolfinx.fem.Function):
                self._u = pyadjoint.create_overloaded_object(u)
            else:
                self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]  # type: ignore[assignment]

        # Cache some objects
        self._lhs = a
        self._rhs = L
        self._preconditioner = P
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options
        self._kind = kind

        # Initialize linear solver
        super().__init__(
            a=a,
            L=L,
            bcs=bcs,
            u=self._u,
            P=P,
            kind=kind,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

    def solve(
        self, annotate: bool = True
    ) -> typing.Tuple[typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]], int, int]:
        """
        Solve the linear problem and return the solution.
        """
        annotate = pyadjoint.annotate_tape({"annotate": annotate})
        if annotate:
            block = LinearProblemBlock(
                self._lhs,  # type: ignore
                self._rhs,  # type: ignore
                bcs=self.bcs,
                u=self.u,
                P=self._preconditioner,
                kind=self._kind,
                petsc_options=self._petsc_options,
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                entity_maps=self._entity_maps,
                ad_block_tag=self.ad_block_tag,
                adjoint_petsc_options=self._adj_options,
                tlm_petsc_options=self._tlm_options,
            )
            tape = pyadjoint.get_working_tape()
            tape.add_block(block)

        out = dolfinx.fem.petsc.LinearProblem.solve(self)
        if annotate:
            if isinstance(out, Function):
                block.add_output(out.create_block_variable())
            else:
                for ui in out:
                    assert isinstance(ui, Function)
                    block.add_output(ui.create_block_variable())
        return out
