"""Tests for ``xarray.indexes.lazy_index``.

A ``DataTree`` opened with ``create_default_indexes=False`` does not get pandas
indexes built for its dimension coordinates, which breaks label-based selection
on inherited child coords. ``create_lazy_default_indices`` installs lazy
wrappers so ``sel`` works without eagerly reading the coord chunk.
"""

from pathlib import Path
from unittest.mock import patch

import xarray as xr
from xarray.indexes import LazyDefaultIndex, PandasIndex, create_lazy_default_indices
from xarray.tests import requires_dask, requires_zarr


@requires_zarr
@requires_dask
def test_create_lazy_default_indices(tmp_path: Path) -> None:
    """
    Tests the main point of LazyIndex:
    * It defers loading until needed
    * it will search the data tree hierarchy for coordinates
    """

    from zarr.storage import LocalStore

    # Setup datatree
    path = tmp_path / "tree.zarr"
    xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                data_vars={"b": (["x"], [5, 6])},
                coords={"x": [10, 20]},
            ),
            "/child": xr.Dataset(data_vars={"a": (["x"], [0, 1])}),
        }
    ).to_zarr(path, mode="w")

    def gets_of(spy, key: str) -> int:
        return sum(1 for c in spy.call_args_list if c.args[1] == key)

    with patch.object(LocalStore, "get", wraps=LocalStore.get, autospec=True) as spy:
        tree = xr.open_datatree(
            LocalStore(path, read_only=True),
            engine="zarr",
            chunks={},
            create_default_indexes=False,
        )
        tree = create_lazy_default_indices(tree)

        root_idx = tree.xindexes["x"]
        child_idx = tree["child"].xindexes["x"]
        assert isinstance(root_idx, LazyDefaultIndex)
        assert isinstance(child_idx, LazyDefaultIndex)
        assert not root_idx.is_materialized
        assert not child_idx.is_materialized

        assert gets_of(spy, "x/c/0") == 0

        assert int(tree.b.sel(x=10).values) == 5
        assert int(tree["child"].a.sel(x=10).values) == 0

        assert root_idx.is_materialized
        assert child_idx.is_materialized
        assert gets_of(spy, "x/c/0") == 1


def test_from_variables_via_set_xindex() -> None:
    ds = xr.Dataset(
        data_vars={"a": (["x"], [10, 20, 30])},
        coords={"x": [1, 2, 3]},
    ).drop_indexes("x")
    indexed = ds.set_xindex("x", LazyDefaultIndex)
    idx = indexed.xindexes["x"]
    assert isinstance(idx, LazyDefaultIndex)
    assert not idx.is_materialized
    assert int(indexed.a.sel(x=2).values) == 20
    assert idx.is_materialized


def test_isel_short_circuits_without_materializing() -> None:
    ds = xr.Dataset(
        data_vars={"a": (["x"], [10, 20, 30])},
        coords={"x": [1, 2, 3]},
    )
    ds = ds.drop_indexes("x")
    ds = create_lazy_default_indices(ds)

    idx = ds.xindexes["x"]
    assert isinstance(idx, LazyDefaultIndex)
    assert not idx.is_materialized

    # Full slice short-circuits at the index level and doesn't read the coord.
    full = ds.isel(x=slice(None))
    assert isinstance(full.xindexes["x"], LazyDefaultIndex)
    assert not idx.is_materialized

    # Sub-slice materialises (the result needs a concrete pandas Index).
    sliced = ds.isel(x=slice(1, 2))
    assert isinstance(sliced.xindexes["x"], PandasIndex)

    # Scalar isel drops the dim from xindexes but still needs the coord value
    # at that position, so it does materialise. (With the PandasIndexingAdapter
    # backing we now use, the coord variable's reads go through the index.)
    scalar = ds.isel(x=0)
    assert "x" not in scalar.xindexes


def test_concat_materializes_and_delegates_to_pandas_index() -> None:
    """``xr.concat`` dispatches to ``indexes[0].concat``; without an override on
    ``LazyDefaultIndex`` this resolves to ``Index.concat`` and raises."""
    ds1 = create_lazy_default_indices(
        xr.Dataset({"a": (["x"], [1, 2])}, coords={"x": [10, 20]}).drop_indexes("x")
    )
    ds2 = create_lazy_default_indices(
        xr.Dataset({"a": (["x"], [3, 4])}, coords={"x": [30, 40]}).drop_indexes("x")
    )

    combined = xr.concat([ds1, ds2], dim="x")

    assert isinstance(combined.xindexes["x"], PandasIndex)
    assert combined.x.values.tolist() == [10, 20, 30, 40]
    assert int(combined.a.sel(x=30).values) == 3


def test_concat_with_pandas_index_first() -> None:
    """When the first dataset carries an eager ``PandasIndex``, xarray dispatches
    to ``PandasIndex.concat`` and passes the lazy index in as a later input, so
    ``PandasIndex.concat`` must accept a ``LazyDefaultIndex`` alongside it."""
    ds1 = xr.Dataset({"a": (["x"], [1, 2])}, coords={"x": [10, 20]})
    ds2 = create_lazy_default_indices(
        xr.Dataset({"a": (["x"], [3, 4])}, coords={"x": [30, 40]}).drop_indexes("x")
    )
    assert isinstance(ds1.xindexes["x"], PandasIndex)
    assert isinstance(ds2.xindexes["x"], LazyDefaultIndex)

    combined = xr.concat([ds1, ds2], dim="x")

    assert isinstance(combined.xindexes["x"], PandasIndex)
    assert combined.x.values.tolist() == [10, 20, 30, 40]
    assert int(combined.a.sel(x=30).values) == 3


def test_binary_op_stays_lazy_via_shared_box_equality() -> None:
    """
    Some operations on a DataTree call align_indexes, which is intended
    [to remove duplicate indexes](https://github.com/pydata/xarray/pull/9555).
    It works by comparing the indexes by equality of values*.
    We have a short circuit that allows us to avoid materializing the index.
    """
    tree = xr.DataTree(xr.Dataset(coords={"x": [1, 2, 3]}).drop_indexes("x"))
    tree["/foo"] = xr.DataTree(xr.Dataset({"bar": ("x", [4, 5, 6])}))
    tree = create_lazy_default_indices(tree)

    idx = tree.xindexes["x"]
    assert isinstance(idx, LazyDefaultIndex)
    assert not idx.is_materialized

    doubled = tree * 2

    assert not idx.is_materialized
    assert not doubled.xindexes["x"].is_materialized
    assert doubled["/foo"].bar.values.tolist() == [8, 10, 12]
