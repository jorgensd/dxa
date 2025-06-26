import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl

from dolfinx_adjoint.petsc_utils import LinearAdjointProblem
from dolfinx_adjoint.types import Function


from .assembly import assemble_compiled_form

try:
    import typing_extensions as typing
except ModuleNotFoundError:
    import typing  # type: ignore[no-redef]

class solver_kwargs(typing.TypedDict):
    ad_block_tag: typing.NotRequired[str]
    """Tag for the block in the adjoint tape."""
    annotate: typing.NotRequired[bool]
    """Whether to annotate the assignment in the adjoint tape."""
    adjoint_petsc_options: typing.NotRequired[dict]

    """PETSc options to pass to the adjoint solver."""
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
        **kwargs: typing.Unpack[solver_kwargs],
    ) -> None:
        self._adjoint_petsc_options = kwargs.pop("adjoint_petsc_options", None)
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

        # Solver for recomputing the linear problem
        self._forward_solver = dolfinx.fem.petsc.LinearProblem(
            self._lhs,
            self._rhs,
            bcs=self._bcs,
            u=self._u,
            P=self._preconditioner,
            petsc_options=self._petsc_options,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            kind=kind,
            entity_maps=self._entity_maps,
        )
        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()
        else:
            self._adjoint_solutions = [u.copy() for u in self._u]
        self._adjoint_solver = LinearAdjointProblem(
            self._compute_adjoint(self._lhs),
            self._rhs,
            bcs=self._bcs,
            u=self._adjoint_solutions,
            P=self._preconditioner,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            kind=kind,
            entity_maps=self._entity_maps,
        )

    # def _create_residual(self)-> ufl.Form:
    #     """Replace the linear problem with a residual of the output function(s)."""

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def _create_replace_map(self, form: ufl.Form) -> dict[Function, Function]:
        """Replace dependencies with latest checkpoint."""
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

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """Prepare for recomputing the block with different control inputs."""

        # Create initial guess for the KSP solver
        # Form independnet compilation would make it possible to use the same KSP for all re-evaluations.
        if isinstance(self._u, Function):
            initial_guess = dolfinx.fem.Function(self._u.function_space, name=self._u.name + "_initial_guess")
        else:
            initial_guess = [dolfinx.fem.Function(u.function_space, name=u.name + "_initial_guess") for u in self._u]

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
        preconditioner = (
            self._replace_coefficients_in_form(self._preconditioner) if self._preconditioner is not None else None
        )
        compiled_lhs = dolfinx.fem.form(
            lhs,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        compiled_rhs = dolfinx.fem.form(
            rhs,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        compiled_preconditioner = (
            dolfinx.fem.form(
                preconditioner,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            if preconditioner is not None
            else None
        )

        # Replace the compiled forms with those with new coefficients.
        self._forward_solver._a = compiled_lhs
        self._forward_solver._L = compiled_rhs
        self._forward_solver._P = compiled_preconditioner
        self._forward_solver.bcs = bcs
        self._forward_solver._u = initial_guess

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> typing.Union[Function, typing.Iterable[Function]]:
        """Recompute the block with the prepared linear problem."""
        return self._forward_solver.solve()

    def _should_compute_boundary_adjoint(
        self, relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]]
    ) -> bool:
        """Determine if the adjoint should be computed with respect to the boundary conditions."""
        bdy = False
        for _, dep in relevant_dependencies:
            if isinstance(dep.output, dolfinx.fem.DirichletBC):
                bdy = True
                break
        return bdy

    @classmethod
    def _compute_adjoint(
        cls, form: typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]]
    ) -> typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]]:
        """
        Compute adjoint of a bilinear form :math:`a(u, v)`, which could be written as a blocked system.
        """
        if isinstance(form, ufl.Form):
            return ufl.adjoint(form)
        else:
            adj_form = []
            for i in range(len(form)):
                assert len(form[i]) == len(form), "Expected a square blocked system."
                adj_form.append([])
                for j in range(len(form[i])):
                    adj_form[i][j] = ufl.adjoint(form[j][i])
            return adj_form

    def compute_residual(self) -> typing.Union[ufl.Form, list[ufl.Form]]:
        """Convert the formulation :math:`a(u, v)=L(v)` into a residual :math:`F(u_b, v) = 0` where
        :math:`u_b` is the solution of the forward problem at the current time and all coefficients are updated.
        """
        # NOTE: Should probably be possible to compile this form once.
        replacement_functions = self.get_outputs()
        if isinstance(self._u, Function):
            assert len(replacement_functions) == 1, (
                f"Expected a single output function, got {len(replacement_functions)}"
            )
            F_form = ufl.action(self._lhs, replacement_functions[0].saved_output) - self._rhs
        else:
            # Blocked formulation (assuming no mixed function-space)
            F_form = []
            assert len(self._u) == len(replacement_functions), (
                f"Expected {len(self._u)} output functions, got {len(replacement_functions)}"
            )
            for i in range(len(self._u)):
                F_form.append(0)
                for j in range(len(self._u)):
                    F_form[-1] += ufl.action(self._lhs[i][j], replacement_functions[j].saved_output) - self._rhs[j]

        # NOTE: Will fail for blocked systems atm
        replacement_map = self._create_replace_map(F_form)
        if isinstance(self._u, Function):
            F_form = ufl.replace(F_form, replacement_map)
        else:
            for j in range(len(F_form)):
                F_form[j] = ufl.replace(F_form[j], replacement_map)
        return F_form

    def prepare_evaluate_adj(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]],
    ) -> typing.Union[ufl.Form, typing.Iterable[ufl.Form]]:
        """Prepare the block for evaluating the adjoint."""

        # Compute (dF/du[v])* for the linear problem.
        F_form = self.compute_residual()
        outputs = [output.saved_output for output in self.get_outputs()]
        if len(outputs) == 1:
            assert isinstance(F_form, ufl.Form)
            dFdu = ufl.derivative(F_form, outputs[0], ufl.TrialFunction(outputs[0].function_space))
        else:
            dFdu = []
            for i in range(len(outputs)):
                dFdu.append([])
                for j in range(len(outputs)):
                    dFdu[-1].append(ufl.derivative(F_form[i], outputs[j], ufl.TrialFunction(outputs[j].function_space)))
        dFdu_adj = self._compute_adjoint(dFdu)

        # Extract dJ/du[v] from the adjoint inputs.
        assert len(adj_inputs) == 1
        adj_rhs = adj_inputs[0]
        dJdu = dolfinx.la.vector(adj_rhs.index_map, adj_rhs.block_size)
        dJdu.array[:] = adj_rhs.array[:].copy()


        # Solve adjoint problem
        compiled_dFdu = dolfinx.fem.form(
            dFdu_adj,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver._b = dJdu.petsc_vec
        self._adjoint_solver.solve()

        return F_form

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    ) -> typing.Union[Function, typing.Iterable[Function]]:
        """Evaluate the adjoint component, i.e. :math:`\frac{\partial Au - b}{\partial c}`."""

        residual = prepared

        c = block_variable.output

        c_rep = block_variable.saved_output
        if isinstance(c, dolfinx.fem.Function):
            dc = ufl.TrialFunction(c.function_space)
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")

        dFdm = -ufl.derivative(residual, c_rep, dc)
        dFdm_adj = ufl.adjoint(dFdm)
        sensitivity = ufl.action(dFdm_adj, self._adjoint_solutions)
        compiled_sensitivity = dolfinx.fem.form(
            sensitivity,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        return assemble_compiled_form(compiled_sensitivity)