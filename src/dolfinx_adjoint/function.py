from __future__ import annotations  # for Python<3.11

import dolfinx
import numpy
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

try:
    import typing_extensions as typing
except ModuleNotFoundError:
    import typing  # type: ignore[no-redef]
from .blocks.function_assigner import FunctionAssignBlock
from .utils import ad_kwargs


def assign(
    value: typing.Union[numpy.inexact, float, int], function: dolfinx.fem.Function, **kwargs: typing.Unpack[ad_kwargs]
):
    """Assign a `value` to a :py:func:`dolfinx_adjoint.Function`.

    Args:
        value: The value to assign to the function.
        function: The function to assign the value to.
        *args: Additional positional arguments to pass to the assign method.
        **kwargs: Additional keyword arguments to pass to the assign method.
    """
    # do not annotate in case of self assignment
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    annotate = annotate_tape(kwargs) and value != function
    if annotate:
        if not isinstance(value, ufl.core.operator.Operator):
            value = create_overloaded_object(value)
        block = FunctionAssignBlock(value, function, ad_block_tag=ad_block_tag)
        tape = get_working_tape()
        tape.add_block(block)

    with stop_annotating():
        if isinstance(value, (numpy.inexact, float, int)):
            function.x.array[:] = value
        elif isinstance(value, dolfinx.fem.Function):
            assert value.function_space == function.function_space, (
                "Function spaces of the value and function must match for assignment."
            )
            function.x.array[:] = value.x.array[:]
        else:
            raise ValueError(f"Unsupported value type for assignment: {type(value)})")
    if annotate:
        block.add_output(function.create_block_variable())
