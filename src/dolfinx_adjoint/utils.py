import typing
import dolfinx
import numpy
import numpy.typing as npt


def function_from_vector(
    V: dolfinx.fem.FunctionSpace,
    vector: typing.Union[
        dolfinx.la.Vector,
        dolfinx.cpp.la.Vector_float32,
        dolfinx.cpp.la.Vector_float64,
        dolfinx.cpp.la.Vector_complex64,
        dolfinx.cpp.la.Vector_complex128,
        dolfinx.cpp.la.Vector_int8,
        dolfinx.cpp.la.Vector_int32,
        dolfinx.cpp.la.Vector_int64,
    ],
) -> dolfinx.fem.Function:
    """Create a new Function from a vector.

    :arg V: The function space
    :arg vector: The vector data.
    """
    if not isinstance(vector, dolfinx.la.Vector):
        vector = dolfinx.la.Vector(vector)
    return dolfinx.fem.Function(V, x=vector)


def gather(vector: dolfinx.la.Vector) -> npt.NDArray[numpy.number]:
    """Gather a vector on all processes."""
    local_size = vector.index_map.size_local * vector.block_size
    comm = vector.index_map.comm
    data = comm.allgather(vector.array[:local_size])
    return numpy.hstack(data)

