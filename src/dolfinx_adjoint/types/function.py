from __future__ import annotations  # for Python<3.11

import dolfinx
import numpy
import ufl
from pyadjoint.overloaded_type import (
    FloatingType,
    create_overloaded_object,
    get_overloaded_class,
    register_overloaded_type,
)
from pyadjoint.tape import annotate_tape, get_working_tape, no_annotations, stop_annotating

from dolfinx_adjoint.blocks.assembly import assemble_compiled_form
from dolfinx_adjoint.utils import function_from_vector, gather

try:
    import typing_extensions as typing
except ModuleNotFoundError:
    import typing  # type: ignore[no-redef]
from dolfinx_adjoint.blocks.function_assigner import FunctionAssignBlock
from dolfinx_adjoint.utils import ad_kwargs


class Function(dolfinx.fem.Function, FloatingType):
    """A class overloading `dolfinx.fem.Function` to support it being used as a control variable
    in the adjoint framework.

    Args:
        V: The function space of the function.
        x: Optional vector to initialize the function with. If not provided, a zero vector is created.
        name: Optional name for the function.
        dtype: Data type of the function values, defaults to `dolfinx.default_scalar_type`.
        **kwargs: Additional keyword arguments to pass to the `pyadjoint.overloaded_type.FloatingType` constructor.

    """

    def __init__(
        self,
        V: dolfinx.fem.FunctionSpace,
        x: typing.Optional[dolfinx.la.Vector] = None,
        name: typing.Optional[str] = None,
        dtype: numpy.dtype = dolfinx.default_scalar_type,
        **kwargs,
    ):
        super(Function, self).__init__(
            V,
            x,
            name,
            dtype,
        )
        FloatingType.__init__(
            self,
            V,
            x,
            name=name,
            dtype=dtype,
            block_class=kwargs.pop("block_class", None),
            _ad_floating_active=kwargs.pop("_ad_floating_active", False),
            _ad_args=kwargs.pop("_ad_args", None),
            output_block_class=kwargs.pop("output_block_class", None),
            _ad_output_args=kwargs.pop("_ad_output_args", None),
            _ad_outputs=kwargs.pop("_ad_outputs", None),
            annotate=kwargs.pop("annotate", True),
            **kwargs,
        )

    @classmethod
    def _ad_init_object(cls, obj):
        return cls(obj.function_space, obj.x, obj.name)

    @no_annotations
    def _ad_create_checkpoint(self):
        checkpoint = create_overloaded_object(self.copy())
        checkpoint.name = self.name + "_checkpoint"
        return checkpoint

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_dot(self, other: typing.Self, options: typing.Optional[dict] = None):
        """Compute the inner product of the current function with ``other`` in the Riesz representation.

        Args:
            other: Function to compute the inner product with.
        """
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return dolfinx.cpp.la.inner_product(self.x._cpp_object, other.x._cpp_object)
        elif riesz_representation == "L2":
            form_compiler_options = options.get("form_compiler_options", None)
            jit_options = options.get("jit_options", None)
            mass = ufl.inner(self, other) * ufl.dx
            compiled_form = dolfinx.fem.form(
                mass,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            return assemble_compiled_form(compiled_form)
        elif riesz_representation == "H1":
            form_compiler_options = options.get("form_compiler_options", None)
            jit_options = options.get("jit_options", None)
            mass_and_stiffness = ufl.inner(self, other) * ufl.dx + ufl.inner(ufl.grad(self), ufl.grad(other)) * ufl.dx
            compiled_form = dolfinx.fem.form(
                mass_and_stiffness,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            return assemble_compiled_form(compiled_form)
        else:
            raise NotImplementedError("Unknown Riesz representation %s" % riesz_representation)

    @no_annotations
    def _ad_mul(self, other: typing.Union[int, float]) -> typing.Self:
        """Multiplication of self with integer or floating value."""
        r = get_overloaded_class(dolfinx.fem.Function)(self.function_space)
        r.x.array[:] = self.x.array * other
        return r

    @no_annotations
    def _ad_add(self, other: typing.Self) -> typing.Self:
        r = get_overloaded_class(dolfinx.fem.Function)(self.function_space)
        r.x.array[:] = self.x.array[:] + other.x.array[:]
        return r

    @no_annotations
    def _ad_convert_riesz(self, value: dolfinx.la.Vector, riesz_map: typing.Optional[dict] = None) -> dolfinx.fem.Function:
        """Convert a vector to a Riesz representation of the function."""
        options = {} if riesz_map is None else riesz_map
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return create_overloaded_object(function_from_vector(self.function_space, value))
        elif riesz_representation == "L2":
            from dolfinx.fem.petsc import assemble_matrix

            from dolfinx_adjoint.petsc_utils import solve_linear_problem

            u = ufl.TrialFunction(self.function_space)
            v = ufl.TestFunction(self.function_space)
            riesz_form = ufl.inner(u, v) * ufl.dx
            compiled_riesz = dolfinx.fem.form(
                riesz_form,
                jit_options=options.get("jit_options", None),
                form_compiler_options=options.get("form_compiler_options", None),
            )
            ret = dolfinx.fem.Function(self.function_space)
            M = assemble_matrix(compiled_riesz)
            M.assemble()
            petsc_options = options.get("petsc_options", {})
            solve_linear_problem(M, ret.x, value, petsc_options=petsc_options)
            M.destroy()
            return ret
        elif riesz_representation == "H1":
            from dolfinx.fem.petsc import assemble_matrix

            from dolfinx_adjoint.petsc_utils import solve_linear_problem

            u = ufl.TrialFunction(self.function_space)
            v = ufl.TestFunction(self.function_space)
            riesz_form = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            compiled_riesz = dolfinx.fem.form(
                riesz_form,
                jit_options=options.get("jit_options", None),
                form_compiler_options=options.get("form_compiler_options", None),
            )
            ret = dolfinx.fem.Function(self.function_space)
            M = assemble_matrix(compiled_riesz)
            M.assemble()
            petsc_options = options.get("petsc_options", {})
            solve_linear_problem(M, ret.x, value, petsc_options=petsc_options)
            M.destroy()
            return ret
        elif callable(riesz_representation):
            return riesz_representation(value)
        else:
            raise NotImplementedError("Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_to_list(m):
        """Convert a function into a list of (global) values."""
        if not hasattr(m, "array"):
            m_v = m.x
        else:
            m_v = m
        m_a = gather(m_v)
        return m_a.tolist()

    def _ad_copy(self):
        """Create a (deep) copy of the function."""
        r = get_overloaded_class(dolfinx.fem.Function)(self.function_space)
        assign(self, r)
        return r

    @staticmethod
    def _ad_assign_numpy(dst: dolfinx.fem.Function, src: numpy.ndarray, offset: int):
        range_begin, range_end = dst.x.index_map.local_range
        range_begin *= dst.x.block_size
        range_end *= dst.x.block_size
        m_a_local = src[offset + range_begin : offset + range_end]
        dst.x.array[: len(m_a_local)] = m_a_local
        offset += dst.x.index_map.size_local * dst.x.block_size
        dst.x.scatter_forward()
        return dst, offset


register_overloaded_type(Function, (dolfinx.fem.Function, Function))


def assign(value: typing.Union[numpy.inexact, float, int], function: Function, **kwargs: typing.Unpack[ad_kwargs]):
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
