import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl

from dolfinx_adjoint.types import Function
from dolfinx_adjoint.utils import ad_kwargs

try:
    import typing_extensions as typing
except ModuleNotFoundError:
    import typing  # type: ignore[no-redef]


class LinearProblemBlock(pyadjoint.Block):
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
        super().__init__(ad_block_tag=kwargs.pop("ad_block_tag", None))
        self._lhs = a
        self._rhs = L
        self._preconditioner = P

        # Create overloaded functions
        if isinstance(u, dolfinx.fem.Function):
            self._u = pyadjoint.create_overloaded_object(u)
        elif u is None:
            try:
                # Extract function space for unknown from the right hand
                # side of the equation.
                self._u = Function(L.arguments()[0].ufl_function_space())
            except AttributeError:
                self.u = [Function(Li.arguments()[0].ufl_function_space()) for Li in L]
        else:
            self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]

        # NOTE: Add mesh and constants as dependencies later on
        try:
            for c in self._lhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
            for c in self._rhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
        except AttributeError:
            raise NotImplementedError("Blocked systems not implemented yet.")
        self._compiled_lhs = dolfinx.fem.form(
            self._lhs,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        self._compiled_rhs = dolfinx.fem.form(
            self._rhs,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options if petsc_options is not None else {}
        self._bcs = bcs if bcs is not None else []

    # def _create_residual(self)-> ufl.Form:
    #     """Replace the linear problem with a residual of the output function(s)."""


    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfin.DirichletBC):
                bcs.append(c_rep)
        return bcs

    
    def _create_replace_map(self, form: ufl.Form) -> dict[Function, Function]:
        """Create a map from the block-dependencies to the corresponding function that is stored on the ``pyadjoint.Tape``."""
        replace_map = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in form.coefficients():
                replace_map[coeff] = block_variable.saved_output
        return replace_map

    def _replace_coefficients_in_form(self, form: ufl.Form) -> ufl.Form:
        """Replace coefficients in the form with saved outputs.
        
        Args:
            form: The UFL form to replace coefficients in.        
        """
        replace_map = self._create_replace_map(form)
        return ufl.replace(form, replace_map)
    
    def prepare_recompute_component(self, inputs, relevant_outputs) -> dolfinx.fem.petsc.LinearProblem:
        """Prepare for recomputing the block with different control inputs."""

        # Create initial guess for the KSP solver
        if isinstance(self._u, Function):
            initial_guess = dolfinx.fem.Function(self._u.function_space, name=self._u.name + "_initial_guess")
        else:
            initial_guess = [dolfinx.fem.Function(u.function_space, name=u.name+"_initial_guess") for u in self._u]

        # Replace values in the DirichletBC if it is dependent on a control
        # NOTE: Currently assume that BCS are control independent.
        bcs = self._bcs
        # for block_variable in self.get_dependencies():
        #     c = block_variable.output
        #     c_rep = block_variable.saved_output

        #     if isinstance(c, dolfinx.fem.DirichletBC):
        #         bcs.append(c_rep)

        # Replace form coefficients with checkpointed values.
        # Loop through the dependencies of the lhs and rhs, check if they are in the respective form
        lhs = self._replace_coefficients_in_form(self._lhs)
        rhs = self._replace_coefficients_in_form(self._rhs)
        print("COEFF (control", rhs.coefficients()[0].x.array)
        preconditioner = self._replace_coefficients_in_form(self._preconditioner) if self._preconditioner is not None else None
        ksp = dolfinx.fem.petsc.LinearProblem(
            lhs, rhs, bcs=bcs, u=initial_guess, P=preconditioner, petsc_options=self._petsc_options,
            form_compiler_options=self._form_compiler_options, jit_options=self._jit_options,
            entity_maps=self._entity_maps
        )        
        return ksp

    def recompute_component(self, inputs, block_variable, idx, prepared) -> typing.Union[Function, typing.Iterable[Function]]:
        """Recompute the block with the prepared linear problem."""
        ksp = prepared
        return ksp.solve()


