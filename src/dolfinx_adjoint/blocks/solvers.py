import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl

from dolfinx_adjoint.petsc_utils import LinearAdjointProblem, solve_linear_problem
from dolfinx_adjoint.types import Function

from .assembly import _create_vector, _SpecialVector, assemble_compiled_form

def create_new_form(form: ufl.Form, dependencies: list[pyadjoint.Block], outputs: list[pyadjoint.block_variable.BlockVariable]) -> tuple[ufl.Form, dict[typing.Union[pyadjoint.Block], Function]]:
    """Replace coefficients in a variational form with placeholder variables,
        either if the variable is an input or output to the variational form.
    
    Args:
        form: The UFL form to replace coefficients in.
        dependencies: List of blocks that contain the dependencies to replace.
        outputs: List of block variables that are outputs of the calculaton.
    Returns:
        The new UFL form and a dictionary mapping each block variable to the coefficient
        that replaces its output in the form.
    """
    replace_map: dict[Function, Function] = {}
    block_to_coeff: dict[pyadjoint.Block, Function] = {}
    for block in dependencies:
        if (coeff:=block.output) in form.coefficients():
            replace_map[coeff] = Function(coeff.function_space, name=coeff.name + "_placeholder")
            block_to_coeff[block] = replace_map[coeff]

    for block_variable in outputs:
        # Create replacement function even if coeff is not in form, as it is used for residual computation.
        if (coeff:=block_variable.output) not in replace_map.keys():
            replace_map[coeff] = Function(coeff.function_space, name=coeff.name + "_placeholder")
        block_to_coeff[block_variable] = replace_map[coeff]
        
    return ufl.replace(form, replace_map), block_to_coeff


def _compute_adjoint(
    form: typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]]
) -> typing.Union[ufl.Form, typing.Sequence[typing.Iterable[ufl.Form]]]:
    """
    Compute adjoint of a bilinear form :math:`a(u, v)`, which could be written as a blocked system.
    """
    if isinstance(form, ufl.Form):
        return ufl.adjoint(form)
    else:
        assert isinstance(form, typing.Iterable)
        adj_form: list[list[ufl.Form]] = []
        tmp_form: list[list[ufl.Form]] = []
        for i, f_i in enumerate(form):
            tmp_form.append([])
            adj_form.append([])
            for j, form_ij in enumerate(f_i):
                tmp_form[i].append(ufl.adjoint(form_ij))
                adj_form[i].append(ufl.adjoint(form_ij))
        for i, f_i in enumerate(tmp_form):
            for j, form_ij in enumerate(f_i):
                adj_form[j][i] = form_ij
        return adj_form

def assign_output_to_form(
    blocks: list[typing.Union[pyadjoint.Block, pyadjoint.block_variable.BlockVariable]],
                 block_to_coeff: dict[pyadjoint.Block, Function]):
    """Assign the `saved_output` of a block variable to the coefficients in a form."""
    for block_variable in blocks:
        form_coeff = block_to_coeff[block_variable]
        form_coeff.x.array[:] = block_variable.saved_output.x.array


def _differentiate(form: typing.Union[ufl.Form, list[ufl.Form]], outputs: list[pyadjoint.block_variable.BlockVariable],
                   block_to_func) -> typing.Union[ufl.Form, list[list[ufl.Form]]]:
    """Compute the derivative of the form with respect to its outputs.
    
    Args:
        form: The UFL form to differentiate.
        outputs: List of block variables that are outputs of the calculation.
        block_to_func: A dictionary mapping block variables to their corresponding functions in the form.   
    """
    if len(outputs) == 1:
        assert isinstance(form, ufl.Form), "Form must be a single UFL form when there is only one output."
        u = block_to_func[outputs[0]]
        dFdu = ufl.derivative(form, u, ufl.TrialFunction(u.function_space))
    else:
        assert len(form) == len(outputs), "Number of outputs must match the number of forms."
        dFdu = []
        u_s = [block_to_func[block] for block in outputs]
        for i, block in enumerate(outputs):
            u = block_to_func[block]
            assert isinstance(form[i], list)
            dFdu.append([])
            for j in range(len(outputs)):
                dFdu[-1].append(ufl.derivative(form[i], outputs[j], ufl.TrialFunction(u_s[j].function_space)))
    return dFdu



class LinearProblemBlock(pyadjoint.Block):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: typing.Union[Function, typing.Iterable[Function]]
    _second_adjoint_solutions: typing.Union[Function, typing.Iterable[Function]]

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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
    ) -> None:
        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
        super().__init__(ad_block_tag=ad_block_tag)

        # Create overloaded functions
        if isinstance(u, dolfinx.fem.Function):
            self._u = pyadjoint.create_overloaded_object(u)
        elif u is None:
            try:
                # Extract function space for unknown from the right hand
                # side of the equation.
                self._u = Function(L.arguments()[0].ufl_function_space())  # type: ignore
            except AttributeError:
                self._u = [Function(Li.arguments()[0].ufl_function_space()) for Li in L]  # type: ignore[union-attr]
        else:
            self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]

        # NOTE: Add mesh and constants as dependencies later on
        try:
            for c in a.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
            for c in L.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
            if P is not None:
                for c in P.coefficients():  # type: ignore
                    self.add_dependency(c, no_duplicates=True)
        except AttributeError:
            raise NotImplementedError("Blocked systems not implemented yet.")
        # Add output of the block
        if isinstance(self._u, Function):
            self.add_output(self._u.create_block_variable())
        else:
            for ui in self._u:
                assert isinstance(ui, Function)
                self.add_output(ui.create_block_variable())

        # Replace coefficients in the form with placeholder variables.
        F, self._block_to_coeff = create_new_form(a-L, self.get_dependencies(), self.get_outputs())
        _a, _L = ufl.system(F)

        if P is not None:
            _P, self._block_to_preconditioner_coeff = create_new_form(P, self.get_dependencies(), self.get_outputs())
        else:
            _P = None
            self._block_to_preconditioner_coeff = None
        self._preconditioner = _P

        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options if petsc_options is not None else {}
        self._bcs = bcs if bcs is not None else []
        # Solver for recomputing the linear problem
        placeholder_outputs = [self._block_to_coeff[block_variable] for block_variable in self.get_outputs()]
        if len(placeholder_outputs) == 1:
            placeholder_outputs = placeholder_outputs[0]
        self._forward_solver = dolfinx.fem.petsc.LinearProblem(
            _a,
            _L,
            bcs=self._bcs,
            u=placeholder_outputs,
            P=_P,
            petsc_options=self._petsc_options,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            kind=kind,
            entity_maps=self._entity_maps,
        )

        self._kind = "nest" if self._forward_solver.A.getType() == "nest" else kind

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()
            self._adjoint_solutions.name  = self._u.name + "_adjoint"
            self._second_adjoint_solutions = self._u.copy()
            self._second_adjoint_solutions.name = self._u.name + "_second_adjoint"
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]


        # Set-up adjoint solver (first and second order adjoint have the same lhs)
        if isinstance(_a, ufl.Form):
            outputs = self.get_outputs()
            assert len(outputs) == 1, "LinearProblemBlock only supports single output blocks."
            # If output is a part of the form, we can replace it with the map, otherwise, we 
            _residual = ufl.action(_a, self._block_to_coeff[outputs[0]])  - _L

        dFdu_adjoint = _compute_adjoint(_differentiate(_residual, self.get_outputs(), self._block_to_coeff))
        self._adjoint_solver = LinearAdjointProblem(
            dFdu_adjoint,
            _L,
            bcs=self._bcs,
            u=self._adjoint_solutions,
            P=self._preconditioner,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            kind=kind,
            entity_maps=self._entity_maps,
        )
        self._compiled_residual = dolfinx.fem.form(
            _residual,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # Compute the derivative of the residual with respect to the inputs.
        self._first_adj_sensitivity = []
        for block  in self.get_dependencies():
            c = self._block_to_coeff[block]
            dc = ufl.TrialFunction(c.function_space)
            dFdm = -ufl.derivative(_residual, c, dc)
            dFdm_adj = ufl.adjoint(dFdm)
            sensitivity = ufl.action(dFdm_adj, self._adjoint_solutions)
            self._first_adj_sensitivity.append(dolfinx.fem.form(
                sensitivity,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
                ))


    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)
        return bcs


    def prepare_recompute_component(self, inputs, relevant_outputs):
        """Prepare for recomputing the block with different control inputs."""


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
        # Replace the compiled forms with those with new coefficients.
        assign_output_to_form(self.get_dependencies(), self._block_to_coeff)
        if self._block_to_preconditioner_coeff is not None:
            assign_output_to_form(self.get_dependencies(), self._block_to_preconditioner_coeff)
        self._forward_solver.bcs = bcs

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]]:
        """Recompute the block with the prepared linear problem."""
        output, _, _ = self._forward_solver.solve()
        return output

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




    def prepare_evaluate_tlm(
        self, inputs, tlm_inputs, relevant_outputs
    ) -> tuple[typing.Union[list[ufl.Form], ufl.Form], dolfinx.fem.Form]:
        F_form = self._compute_residual()
        dFdu_compiled = dolfinx.fem.form(
            self._compute_residual_derivative(),  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        return F_form, dFdu_compiled

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None) -> dolfinx.fem.Function:
        """Solve the TLM equation for the block variable.

        .. math::

            \frac{\\partial F}{\\partial u} \frac{\\partial u}{\\partial m} = \frac{\\partial F}{\\partial m}

        """
        # FIXME: Think about blocks later
        F, dFdu = prepared

        V = self.get_outputs()[idx].output.function_space

        # FIXME: DirichletBC not block variable yet. Required later on. Currently all bcs should be homogenized
        bcs = []
        for bc in self._bcs:
            bcs.append(bc)
        dFdm = ufl.ZeroBaseForm((ufl.TestFunction(V),))
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            # c = block_variable.output
            c_rep = block_variable.saved_output
            if tlm_value is None:
                continue

            dFdm += ufl.derivative(-F, c_rep, tlm_value)

        if isinstance(dFdm, float):
            v = dFdu.arguments()[0]
            dFdm = ufl.ZeroBaseForm((v,))

        dFdm = ufl.algorithms.expand_derivatives(dFdm)
        dFdm_compiled = dolfinx.fem.form(
            dFdm,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        dudm = dolfinx.fem.Function(V, name="du_dm_tlm_linearblock")
        A_tlm = dolfinx.fem.petsc.assemble_matrix(dFdu, bcs=bcs)
        A_tlm.assemble()
        b_tlm = dolfinx.fem.create_vector(dFdm_compiled)
        b_tlm.array[:] = 0.0
        dolfinx.fem.petsc.assemble_vector(b_tlm.petsc_vec, dFdm_compiled)

        if bcs is not None:
            # This system should never be "blocked"
            dolfinx.fem.petsc.apply_lifting(b_tlm.petsc_vec, [dFdu], bcs=[bcs], alpha=0)
            dolfinx.la.petsc._ghost_update(b_tlm.petsc_vec, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore [attr-defined]
            for bc in bcs:
                bc.set(b_tlm.array, alpha=0)
        else:
            dolfinx.la.petsc._ghost_update(b_tlm, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore [attr-defined]
        solve_linear_problem(A_tlm, dudm.x, b_tlm, petsc_options=self._tlm_petsc_options)
        return dudm

    def prepare_evaluate_adj(
        self,
        inputs: typing.Sequence[Function],
        adj_inputs: typing.Sequence[dolfinx.la.Vector],
        relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]],
    ) -> typing.Union[ufl.Form, typing.Iterable[ufl.Form]]:
        """Prepare the block for evaluating the adjoint."""

        # Extract dJ/du[v] from the adjoint inputs.
        assert len(adj_inputs) == 1
        adj_rhs = adj_inputs[0]
        dJdu = dolfinx.la.vector(adj_rhs.index_map, adj_rhs.block_size)
        dJdu.array[:] = adj_rhs.array[:].copy()

        # Update coefficients in the form with saved outputs
        assign_output_to_form(self.get_dependencies(), self._block_to_coeff)

        # Solve adjoint problem
        self._adjoint_solver._b = dJdu.petsc_vec
        self._adjoint_solver._u = self._adjoint_solutions  # type: ignore[assignment]
        self._adjoint_solver.solve()

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    ) -> typing.Union[_SpecialVector, typing.Iterable[_SpecialVector]]:
        """Evaluate the adjoint component, i.e. :math:`\frac{\\partial Au - b}{\\partial c}`."""

        # Get form and assign coefficients
        sensitivty = self._first_adj_sensitivity[idx]
        assign_output_to_form(self.get_dependencies(), self._block_to_coeff)

        # Compute assembly
        vec = _create_vector(sensitivty)
        assemble_compiled_form(sensitivty, tensor=vec)
        return vec

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        # First fetch all relevant values
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs if output is not None]
        if (hessian_inputs is None) or (len(tlm_output) == 0):
            return

        # Using the equation Form we derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = self._compute_residual_derivative()
        assert len(outputs) == 1, "Hessian computation only implemented for single output blocks."
        assert len(tlm_output) == 1, "Hessian computation only implemented for single TLM output blocks."
        d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_form, outputs[0].saved_output, tlm_output[0]))

        # bdy = self._should_compute_boundary_adjoint(relevant_dependencies)
        assert len(hessian_inputs) == 1, "Hessian computation only implemented for single hessian input blocks."

        # Assemble right hand side of second order adjoint equation
        b_form = d2Fdu2 if d2Fdu2.empty() else ufl.action(ufl.adjoint(d2Fdu2), self._adjoint_solutions)
        for bo in self.get_dependencies():
            c = bo.output
            tlm_input = bo.tlm_value
            if tlm_input is None:
                continue
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")

        b = dolfinx.la.vector(hessian_inputs[0].index_map, hessian_inputs[0].block_size)
        b.array[:] = 0.0
        if not b_form.empty():
            compiled_soa_rhs = dolfinx.fem.form(
                ufl.algorithms.expand_derivatives(b_form),
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            dolfinx.fem.petsc.assemble_vector(b.petsc_vec, compiled_soa_rhs)
            b.array.scatter_reverse(dolfinx.la.InsertMode.ADD)
            b.array[:] *= -1
        b.array[:] += hessian_inputs[0].array

        # Compile SOA LHS
        dFdu_adj = dolfinx.fem.form(
            ufl.adjoint(dFdu_form),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # Solve adjoint problem
        self._adjoint_solver._a = dFdu_adj
        self._adjoint_solver._b = b.petsc_vec
        self._adjoint_solver._u = self._second_adjoint_solutions
        self._adjoint_solver.solve()
        return self._compute_residual(), self._adjoint_solutions, self._second_adjoint_solutions

    def evaluate_hessian_component(
        self,
        inputs,
        hessian_inputs,
        adj_inputs,
        block_variable,
        idx,
        relevant_dependencies,
        prepared=None,
    ):
        c = block_variable.output

        F_form, adj_sol, adj_sol2 = prepared

        outputs = self.get_outputs()
        assert len(outputs) == 1, "Hessian computation only implemented for single output blocks."
        tlm_output = outputs[0].tlm_value

        c_rep = block_variable.saved_output

        # If m = DirichletBC then d^2F(u,m)/dm^2 = 0 and d^2F(u,m)/dudm = 0,
        # so we only have the term dF(u,m)/dm * adj_sol2
        if isinstance(c, dolfinx.fem.DirichletBC):
            raise NotImplementedError("Hessian computation for DirichletBC control not implemented yet.")

        if isinstance(c_rep, dolfinx.fem.Constant):
            raise NotImplementedError("Hessian computation for Constant control not implemented yet.")
            # mesh = extract_mesh_from_form(F_form)
            # W = c._ad_function_space(mesh)

        elif isinstance(c, dolfinx.mesh.Mesh):
            raise NotImplementedError("Hessian computation for Mesh control not implemented yet.")
            # X = dolfin.SpatialCoordinate(c)
            # W = c._ad_function_space()
        else:
            assert isinstance(c, dolfinx.fem.Function)
            W = c.function_space

        dc = ufl.TestFunction(W)
        form_adj = ufl.action(F_form, adj_sol)
        form_adj2 = ufl.action(F_form, adj_sol2)
        if isinstance(c, dolfinx.mesh.Mesh):
            raise NotImplementedError("Hessian computation for Mesh control not implemented yet.")
            # dFdm_adj = ufl.derivative(form_adj, X, dc)
            # dFdm_adj2 = ufl.derivative(form_adj2, X, dc)
        else:
            # Assume Function
            dFdm_adj = ufl.derivative(form_adj, c_rep, dc)
            dFdm_adj2 = ufl.derivative(form_adj2, c_rep, dc)

        # TODO: Old comment claims this might break on split. Confirm if true or not.
        d2Fdudm = ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, outputs[0].saved_output, tlm_output))

        d2Fdm2 = 0
        # We need to add terms from every other dependency
        # i.e. the terms d^2F/dm_1dm_2
        for _, bv in relevant_dependencies:
            c2 = bv.output
            c2_rep = bv.saved_output

            if isinstance(c2, dolfinx.fem.DirichletBC):
                continue
            tlm_input = bv.tlm_value
            if tlm_input is None:
                continue

            if c2 == self._u and not self.linear:
                continue

            # TODO: If tlm_input is a Sum, this crashes in some instances?
            if isinstance(c2_rep, dolfinx.mesh.Mesh):
                X = ufl.SpatialCoordinate(c2_rep)
                d2Fdm2 += ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, X, tlm_input))
            else:
                d2Fdm2 += ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, c2_rep, tlm_input))

        hessian_form = ufl.algorithms.expand_derivatives(d2Fdm2 + dFdm_adj2 + d2Fdudm)

        compiled_hessian = dolfinx.fem.form(
            hessian_form,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        hessian_output = assemble_compiled_form(compiled_hessian)
        hessian_output.array[:] *= -1.0
        return hessian_output
