import typing

import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl

from dolfinx_adjoint.types import Function

from .blocks.solvers import LinearProblemBlock
from .utils import ad_kwargs


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
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
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        entity_maps: typing.Optional[dict[dolfinx.mesh.Mesh, npt.NDArray[np.int32]]] = None,
        **kwargs: typing.Unpack[ad_kwargs],
    ) -> None:
        self.ad_block_tag = kwargs.pop("ad_block_tag", None)

        if u is None:
            try:
                # Extract function space for unknown from the right hand
                # side of the equation.
                self._u = Function(L.arguments()[0].ufl_function_space())
            except AttributeError:
                self._u = [Function(Li.arguments()[0].ufl_function_space()) for Li in L]
        else:
            if isinstance(u, dolfinx.fem.Function):
                self._u = pyadjoint.create_overloaded_object(u)
            else:
                self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]

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
        dolfinx.fem.petsc.LinearProblem.__init__(
            self, a, L, bcs, self._u, P, kind, petsc_options, form_compiler_options, jit_options, entity_maps
        )

    def solve(self, annotate: bool = True) -> typing.Union[Function, typing.Iterable[Function]]:
        """
        Solve the linear problem and return the solution.
        """
        annotate = pyadjoint.annotate_tape({"annotate": annotate})
        if annotate:
            # FIXME: Decide what objects should be passed in here.
            block = LinearProblemBlock(
                self._lhs,
                self._rhs,
                bcs=self.bcs,
                u=self.u,
                P=self._preconditioner,
                kind=self._kind,
                petsc_options=self._petsc_options,
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                entity_maps=self._entity_maps,
                ad_block_tag=self.ad_block_tag,
            )
            tape = pyadjoint.get_working_tape()
            tape.add_block(block)
        out = dolfinx.fem.petsc.LinearProblem.solve(self)
        if annotate:
            if isinstance(out, dolfinx.fem.Function):
                block.add_output(out.create_block_variable())
            else:
                for ui in out:
                    block.add_output(ui.create_block_variable())
        return out
