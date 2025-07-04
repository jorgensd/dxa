import ufl
import pyadjoint
from dolfinx_adjoint.types import Function
import typing
import dolfinx

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


def compute_adjoint(
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


def assign_tlm_value_to_form(
    blocks: list[typing.Union[pyadjoint.Block, pyadjoint.block_variable.BlockVariable]],
    block_to_coeff: dict[pyadjoint.Block, Function]):
    """Assign the `tlm_value` of a block variable to the coefficients in a form."""
    for block_variable in blocks:
        form_coeff = block_to_coeff[block_variable]
        if block_variable.tlm_value is not None:
            form_coeff.x.array[:] = block_variable.tlm_value.x.array
        else:
            form_coeff.x.array[:] = 0.0

def output_derivative(form: typing.Union[ufl.Form, list[ufl.Form]], outputs: list[pyadjoint.block_variable.BlockVariable],
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


def input_adj_derivative(F: typing.Union[ufl.Form, list[ufl.Form]], inputs: list[pyadjoint.block_variable.BlockVariable],
                          block_to_func: dict[typing.Union[pyadjoint.Block, pyadjoint.block_variable.BlockVariable], Function], lmdba: typing.Union[Function, list[Function]]) -> list[ufl.Form]:
    """Compute the adjoint of the derivative of the residual form with respect to its inputs, i.e.
            :math:`\\left(\\frac{\\partial F_i}{\\partial m_j}\\right)^*\lambda`,
    where :math:`F_i` is the ith residual form, :math:`m` are the inputs and :math:`\\lambda`` the adjoint(s).
    
    Args:
        F: Residual or list of residaul forms to differentiate.
        inputs: List of block variables that are inputs to the calculation.
        block_to_func: A dictionary mapping block variables to their corresponding functions in the form.   
        lmbda: The adjoint variable(s) to apply the derivative to
    """
    adj_sensitivity = []
    if isinstance(F, ufl.Form):
        assert isinstance(lmdba, dolfinx.fem.Function)
        for block in inputs:
            c = block_to_func[block]
            dc = ufl.TrialFunction(c.function_space)
            dFdm = - ufl.derivative(F, c, dc)
            dFdm_adj = ufl.adjoint(dFdm)
            form = ufl.action(dFdm_adj, lmdba)
            adj_sensitivity.append(form)

    else:
        assert isinstance(F, list)
        for block in inputs:
            c = block_to_func[block]
            dc = ufl.TrialFunction(c.function_space)
            form = ufl.ZeroBaseForm((dc, ))
            for F_i, lmbda_i in zip(F, lmdba, strict=True):
                assert isinstance(F_i, list)
                dFdm = - ufl.derivative(F_i, c, dc)
                dFdm_adj = ufl.adjoint(dFdm)
                form += ufl.action(dFdm_adj, lmbda_i)
            adj_sensitivity.append(form)
    return adj_sensitivity



def input_tlm_derivative(F: typing.Union[ufl.Form, list[ufl.Form]], inputs: list[pyadjoint.block_variable.BlockVariable],
                          block_to_func: dict[typing.Union[pyadjoint.Block, pyadjoint.block_variable.BlockVariable], Function],
                          tlm_to_func: dict[typing.Union[pyadjoint.Block, pyadjoint.block_variable.BlockVariable], Function]
                          ) -> typing.Union[ufl.Form, list[ufl.Form]]:
    """Compute the derivative of the residual form with respect to its inputs applied to the tlm direction, i.e.
            :math:`\\left(\\frac{\\partial F_i}{\\partial m_j}\\right) t`,
    where :math:`F_i` is the ith residual form, :math:`m` are the inputs and :math:`t`` is the TLM input(s).
    
    Args:
        F: Residual or list of residaul forms to differentiate.
        inputs: List of block variables that are inputs to the calculation.
        block_to_func: A dictionary mapping block variables to their corresponding functions in the form.   
        t: The tlm inputs
    """
    if isinstance(F, ufl.Form):
        adj_sensitivity = ufl.ZeroBaseForm((F.arguments()[0],))
        for block in inputs:
            c = block_to_func[block]
            dc = ufl.TrialFunction(c.function_space)
            dFdm = - ufl.derivative(F, c, dc)
            form = ufl.action(dFdm, tlm_to_func[block])
            adj_sensitivity += form

    else:
        assert isinstance(F, list)
        adj_sensitivity = []
        for block in inputs:
            c = block_to_func[block]
            dc = ufl.TrialFunction(c.function_space)
            form = ufl.ZeroBaseForm((dc, ))
            for F_i, tlm_i in zip(F, tlm_to_func[block], strict=True):
                assert isinstance(F_i, list)
                dFdm = - ufl.derivative(F_i, c, dc)
                form += ufl.action(dFdm, tlm_i)    
            adj_sensitivity.append(form)
    return adj_sensitivity
