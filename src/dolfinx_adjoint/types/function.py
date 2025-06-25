from typing import Optional

import dolfinx
import numpy
from pyadjoint.overloaded_type import (
    FloatingType,
    create_overloaded_object,
    get_overloaded_class,
    register_overloaded_type,
)
from pyadjoint.tape import no_annotations

from dolfinx_adjoint import assign
from dolfinx_adjoint.utils import function_from_vector, gather


class Function(dolfinx.fem.Function, FloatingType):
    def __init__(
        self,
        V: dolfinx.fem.FunctionSpace,
        x: Optional[dolfinx.la.Vector] = None,
        name: Optional[str] = None,
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
        dolfinx.fem.Function.__init__(self, V, x, name, dtype)

    @classmethod
    def _ad_init_object(cls, obj):
        return cls(obj.function_space, obj.x)

    @no_annotations
    def _ad_create_checkpoint(self):
        return create_overloaded_object(self.copy())

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    @no_annotations
    def _ad_convert_type(self, value: dolfinx.la.Vector, options=None):
        """Convert a vector to a Riesz representation of the function."""

        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return create_overloaded_object(function_from_vector(self.function_space, value))
        # elif riesz_representation == "L2":
        #     ret = Function(self.function_space())
        #     u = dolfin.TrialFunction(self.function_space())
        #     v = dolfin.TestFunction(self.function_space())
        #     M = dolfin.assemble(dolfin.inner(u, v) * dolfin.dx)
        #     linalg_solve(M, ret.vector(), value)
        #     return ret
        # elif riesz_representation == "H1":
        #     ret = Function(self.function_space())
        #     u = dolfin.TrialFunction(self.function_space())
        #     v = dolfin.TestFunction(self.function_space())
        #     M = dolfin.assemble(
        #         dolfin.inner(u, v) * dolfin.dx + dolfin.inner(
        #             dolfin.grad(u), dolfin.grad(v)) * dolfin.dx)
        #     linalg_solve(M, ret.vector(), value)
        #     return ret
        # elif callable(riesz_representation):
        #     return riesz_representation(value)
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
