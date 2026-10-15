"""Xarray index objects for label-based selection and alignment of Dataset /
DataArray objects.

"""

from xarray.core.coordinate_transform import CoordinateTransform
from xarray.core.indexes import (
    CoordinateTransformIndex,
    Index,
    PandasIndex,
    PandasMultiIndex,
)
from xarray.indexes.lazy_index import (
    LazyDefaultIndex,
    create_lazy_default_indices,
)
from xarray.indexes.nd_point_index import NDPointIndex, TreeAdapter
from xarray.indexes.range_index import RangeIndex

__all__ = [
    "CoordinateTransform",
    "CoordinateTransformIndex",
    "Index",
    "LazyDefaultIndex",
    "NDPointIndex",
    "PandasIndex",
    "PandasMultiIndex",
    "RangeIndex",
    "TreeAdapter",
    "create_lazy_default_indices",
]
