from __future__ import annotations  # for Python<3.11

from typing import Unpack

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


class assign_kwargs(typing.TypedDict):
    ad_block_tag: typing.NotRequired[str]
    """Tag for the block in the adjoint tape."""
    annotate: typing.NotRequired[bool]
    """Whether to annotate the assignment in the adjoint tape."""


def assign(
    value: typing.Union[numpy.inexact, float, int], function: dolfinx.fem.Function, **kwargs: Unpack[assign_kwargs]
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
