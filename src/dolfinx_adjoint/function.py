from typing import Optional, TypedDict

import dolfinx
import numpy
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.function_assigner import FunctionAssignBlock


class assign_kwargs(TypedDict):
    ad_block_tag: Optional[str]
    """Tag for the block in the adjoint tape."""
    annotate: bool
    """Whether to annotate the assignment in the adjoint tape."""


def assign(value: numpy.inexact, function: dolfinx.fem.Function, **kwargs: assign_kwargs):
    """Assign a `value` to a `dolfinx.fem.Function`.

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
        if isinstance(value, numpy.inexact):
            function.x.array[:] = value

    if annotate:
        block.add_output(function.create_block_variable())
