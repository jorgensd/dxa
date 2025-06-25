from mpi4py import MPI

import dolfinx
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.assembly import AssembleBlock


def assemble_scalar(form: ufl.Form, **kwargs):
    """Assemble as scalar value from a form.

    Args:
        form: Symbolic form (UFL) to assemble.
        kwargs: Keyword arguments to pass to the assembly routine.
            Includes ``"ad_block_tag"`` to tag the block in the adjoint tape,
            ``"annotate"`` to control whether the assembly is annotated in the adjoint tape,
            ``"jit_options"`` for JIT compilation options,
            and ``"form_compiler_options"`` for form compiler options and ``"entity_map"`` for assembling with Arguments
            and coefficients form meshes that has some relation.
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)

    annotate = annotate_tape(kwargs)
    with stop_annotating():
        compiled_form = dolfinx.fem.form(
            form,
            jit_options=kwargs.pop("jit_options", None),
            form_compiler_options=kwargs.pop("form_compiler_options", None),
            entity_maps=kwargs.pop("entity_maps", None),
        )

        local_output = dolfinx.fem.assemble_scalar(compiled_form)
        comm = compiled_form.mesh.comm
        output = comm.allreduce(local_output, op=MPI.SUM)
        assert isinstance(output, float)

    output = create_overloaded_object(output)

    if annotate:
        block = AssembleBlock(form, ad_block_tag=ad_block_tag)

        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.block_variable)

    return output
