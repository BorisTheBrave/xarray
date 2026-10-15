"""Lazy default-index creation for xarray Datasets and DataTrees.

When a zarr archive is opened with ``create_default_indexes=False``, xarray
skips building pandas indexes for dimension coordinates. Avoiding the read of
the coordinate arrays makes ``open_datatree`` faster and lighter, but it
breaks label-based selection. In particular, ``DataTree`` child nodes inherit
dim coordinates from their parent without inheriting an index, so
``child.var.sel(x=...)`` raises even though ``parent.var.sel(x=...)`` works.

``create_lazy_default_indices`` is the analogue of xarray's internal
``_maybe_create_default_indexes``: for every dim coordinate without an index,
it installs a :class:`LazyDefaultIndex` that defers
``create_default_index_implicit`` until the first index operation.
"""

from __future__ import annotations

import copy as _copy
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Self, overload

import numpy as np
import pandas as pd

from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.indexes import (
    Index,
    PandasIndex,
    create_default_index_implicit,
)
from xarray.core.indexing import (
    IndexSelResult,
    PandasIndexingAdapter,
)
from xarray.core.types import JoinOptions
from xarray.core.utils import is_scalar
from xarray.core.variable import IndexVariable, Variable


class LazyDefaultIndex(Index):
    """An :class:`xarray.indexes.Index` that defers its real construction.

    Wraps a 1-D dimension coordinate variable. The underlying
    :class:`PandasIndex` is built via ``create_default_index_implicit`` on the
    first method call that requires it, then cached. All :class:`Index` API
    methods forward to the materialised index.
    """

    def __init__(
        self,
        variable: Variable,
        all_variables: Iterable[Hashable] | None = None,
    ):
        if variable.ndim != 1:
            raise ValueError(
                "LazyDefaultIndex requires a 1-D dimension coordinate variable; "
                f"got dims={variable.dims!r}"
            )
        self._variable = variable
        self._all_variables: list[Hashable] | None = (
            list(all_variables) if all_variables is not None else None
        )
        # box is a one-element list so that copies of this index share materialisation state
        self._real_box: list[PandasIndex | None] = [None]

    @property
    def _real(self) -> PandasIndex | None:
        return self._real_box[0]

    @_real.setter
    def _real(self, value: PandasIndex | None) -> None:
        self._real_box[0] = value

    @property
    def dim(self) -> Hashable:
        return self._variable.dims[0]

    @property
    def is_materialized(self) -> bool:
        return self._real is not None

    @property
    def index(self) -> pd.Index:
        # ``PandasIndex.concat`` reads ``idx.index`` off every input. When the
        # first dataset carries an eager ``PandasIndex``, xarray dispatches to
        # ``PandasIndex.concat`` and a lazy index arrives here as a later input.
        return self._materialize().index

    @property
    def coord_dtype(self) -> Any:
        # Also read by ``PandasIndex.concat`` (see :attr:`index`).
        return self._materialize().coord_dtype

    def _materialize(self) -> PandasIndex:
        if self._real is None:
            self._real, _ = create_default_index_implicit(
                self._variable, self._all_variables
            )
        return self._real

    @classmethod
    def _wrap_materialized(cls, real: PandasIndex) -> Self:
        """Wrap an already-materialized :class:`PandasIndex`.

        Used by methods like :meth:`join` that produce a new
        :class:`PandasIndex` but need to return a ``LazyDefaultIndex`` to
        satisfy ``Self`` typing on overrides.
        """
        obj = cls.__new__(cls)
        obj._variable = Variable((real.dim,), np.asarray(real.index))
        obj._all_variables = None
        obj._real_box = [real]
        return obj

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> LazyDefaultIndex:
        if len(variables) != 1:
            raise ValueError(
                "LazyDefaultIndex only accepts one variable, "
                f"found {len(variables)} variables"
            )
        _, var = next(iter(variables.items()))
        return cls(var, options.get("all_variables"))

    @classmethod
    def concat(
        cls,
        indexes: Sequence[Index],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> PandasIndex:
        # Same as PandasIndex.concat. We've set up LazyDefaultIndex to
        # be sufficiently similar to PandasIndex that this will materialize
        # each lazy index, then concatenate them.
        return PandasIndex.concat(indexes, dim, positions)

    def sel(self, labels: dict[Any, Any], **kwargs: Any) -> IndexSelResult:
        return self._materialize().sel(labels, **kwargs)

    def isel(
        self,
        indexers: Mapping[Any, int | slice | np.ndarray | Variable],
    ) -> Index | None:
        # Often, the index isn't needed for isel, so we short-circuit here, similar to PandasIndex.isel
        indxr = indexers[self.dim]
        if isinstance(indxr, Variable):
            if indxr.dims != (self.dim,):
                return None
            indxr = indxr.data
        if not isinstance(indxr, slice) and is_scalar(indxr):
            return None
        if isinstance(indxr, slice) and indxr == slice(None):
            return self

        return self._materialize().isel(indexers)

    def equals(
        self,
        other: Index,
        *,
        exclude: frozenset[Hashable] | None = None,
    ) -> bool:
        if other is self:
            return True
        if isinstance(other, LazyDefaultIndex):
            # No need to materialize if they share
            # the same underlying data.
            if other._real_box is self._real_box:
                return True
            other = other._materialize()
        return self._materialize().equals(other, exclude=exclude)

    def join(self, other: Self | PandasIndex, how: JoinOptions = "inner") -> Self:
        if isinstance(other, LazyDefaultIndex):
            other = other._materialize()
        joined = self._materialize().join(other, how=how)
        return type(self)._wrap_materialized(joined)

    def reindex_like(
        self, other: Self | PandasIndex, **kwargs: Any
    ) -> dict[Hashable, Any]:
        if isinstance(other, LazyDefaultIndex):
            other = other._materialize()
        return self._materialize().reindex_like(other, **kwargs)

    def to_pandas_index(self) -> pd.Index:
        return self._materialize().to_pandas_index()

    def roll(self, shifts: Mapping[Any, int]) -> Index | None:
        if shifts[self.dim] % self._variable.shape[0] == 0:
            return self
        return self._materialize().roll(shifts)

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Index:
        if self._real is not None:
            return self._real.rename(name_dict, dims_dict)
        # Stay lazy: rename the wrapped variable so materialisation reflects new names.
        old_dims = self._variable.dims
        new_dims = tuple(dims_dict.get(d, d) for d in old_dims)
        if new_dims == old_dims:
            new_var = self._variable
        else:
            new_var = Variable(
                new_dims,
                self._variable._data,
                self._variable.attrs,
                self._variable.encoding,
            )
        new_all = (
            [name_dict.get(n, n) for n in self._all_variables]
            if self._all_variables is not None
            else None
        )
        return LazyDefaultIndex(new_var, new_all)

    def create_variables(
        self,
        variables: Mapping[Any, Variable] | None = None,
    ) -> dict[Hashable, Variable]:
        # Once materialised, delegate to the real PandasIndex so the coord
        # variable is backed by a normal PandasIndexingAdapter (and not our
        # lazy wrapper) from that point on.
        if self._real is not None:
            return self._real.create_variables(variables)

        # Mirrors PandasIndex.create_variables line by line, swapping in a
        # LazyIndexingAdapter over ``self`` instead of a PandasIndexingAdapter
        # over a materialised ``pd.Index``.
        name = self.dim
        attrs: Mapping[Hashable, Any] | None
        encoding: Mapping[Hashable, Any] | None

        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        data = LazyIndexingAdapter(self)
        var = IndexVariable(
            self.dim, data, attrs=attrs, encoding=encoding, fastpath=True
        )
        return {name: var}

    def should_add_coord_to_array(
        self,
        name: Hashable,
        var: Variable,
        dims: set[Hashable],
    ) -> bool:
        return all(d in dims for d in var.dims)

    def __getitem__(self, indexer: Any) -> Index:
        return self._materialize()[indexer]

    def _copy(
        self,
        deep: bool = True,
        memo: dict[int, Any] | None = None,
    ) -> LazyDefaultIndex:
        new = LazyDefaultIndex.__new__(LazyDefaultIndex)
        if deep:
            new._variable = _copy.deepcopy(self._variable, memo)
            new._all_variables = _copy.deepcopy(self._all_variables, memo)
        else:
            new._variable = self._variable
            new._all_variables = self._all_variables
        # share materialisation state
        # This is ok even for deep copies as the index (once materialized) is immutable
        new._real_box = self._real_box
        return new

    def __repr__(self) -> str:
        if self._real is None:
            return f"{type(self).__name__}(unmaterialized, dim={self.dim!r})"
        return f"{type(self).__name__}({self._real!r})"


class LazyIndexingAdapter(PandasIndexingAdapter):
    """Variable backing for :class:`LazyDefaultIndex`.

    This wraps a PandasIndexingAdapter that is materialized on the first read.
    It also inherits from PandasIndexingAdapter to trick IndexVariable's constructor to pass it through unchanged.
    """

    __slots__ = ("_index", "_proxy")

    def __init__(self, lazy_index: LazyDefaultIndex):
        # Skip ``PandasIndexingAdapter.__init__`` deliberately — it would call
        # ``safe_cast_to_index`` on the data and force materialisation.
        self._index = lazy_index
        self._proxy: PandasIndexingAdapter | None = None

    def _materialize(self) -> PandasIndexingAdapter:
        if self._proxy is None:
            real = self._index._materialize()
            self._proxy = PandasIndexingAdapter(real.index, dtype=real.coord_dtype)
        return self._proxy

    @property
    def array(self) -> pd.Index:  # type: ignore[override]
        return self._materialize().array

    @property
    def _dtype(self) -> Any:  # type: ignore[override]
        if self._proxy is not None:
            return self._proxy._dtype
        return self._index._variable.dtype

    @property
    def shape(self) -> tuple[int, ...]:  # type: ignore[override]
        if self._proxy is not None:
            return self._proxy.shape
        return self._index._variable.shape

    def __getitem__(self, indexer: Any) -> Any:
        return self._materialize()[indexer]

    def _oindex_get(self, indexer: Any) -> Any:
        return self._materialize()._oindex_get(indexer)

    def _vindex_get(self, indexer: Any) -> Any:
        return self._materialize()._vindex_get(indexer)

    def __repr__(self) -> str:
        if self._proxy is None:
            return f"{type(self).__name__}(unresolved, dim={self._index.dim!r})"
        return f"{type(self).__name__}({self._proxy!r})"


def _lazy_indices_for_dataset(ds: Dataset) -> Dataset:
    """Install :class:`LazyDefaultIndex` on every unindexed 1-D dim coord of ``ds``."""
    to_index = {
        name: coord.variable
        for name, coord in ds.coords.items()
        if coord.dims == (name,) and name not in ds.xindexes
    }
    if not to_index:
        return ds

    all_names = list(to_index)
    for name in all_names:
        ds = ds.set_xindex([name], LazyDefaultIndex, all_variables=all_names)
    return ds


@overload
def create_lazy_default_indices(obj: Dataset) -> Dataset: ...
@overload
def create_lazy_default_indices(obj: DataTree) -> DataTree: ...
def create_lazy_default_indices(
    obj: Dataset | DataTree,
) -> Dataset | DataTree:
    """Add lazy default indexes to every unindexed dimension coordinate.

    Mirrors xarray's internal ``_maybe_create_default_indexes`` (the helper run
    when ``open_datatree(create_default_indexes=True)``), but installs
    :class:`LazyDefaultIndex` wrappers instead of materialising
    :class:`PandasIndex` objects up front. The actual pandas index is built on
    the first index operation that needs it (e.g. ``sel``).

    Args:
        obj: A :class:`Dataset` or :class:`DataTree`. For a tree, the function
            is applied to every node.

    Returns:
        A new ``Dataset``/``DataTree`` with lazy indexes installed for every
        unindexed 1-D dimension coordinate.
    """
    if isinstance(obj, DataTree):
        return obj.map_over_datasets(_lazy_indices_for_dataset)
    return _lazy_indices_for_dataset(obj)
