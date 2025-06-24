import dolfinx
from typing import Optional
from pyadjoint.overloaded_type import (
    FloatingType,
    create_overloaded_object,
    register_overloaded_type,
)
from pyadjoint.tape import no_annotations
import numpy


@register_overloaded_type
class Function(FloatingType, dolfinx.fem.Function):
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
        FloatingType.__init__(self, V, x,name=name, dtype=dtype,
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
