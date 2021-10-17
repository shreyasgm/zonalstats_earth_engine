import pytest
import numpy as np
import pandas as pd
from projections import raster


@pytest.mark.parametrize(
    "min_,max_,array,expected",
    (
        (2, 3, [0, 1, 2, 3, 4, 5, 6], [2, 3]),
        (1.8, 3, [1, 2, 3, 4, 5, 6], [1, 2, 3]),
        (1.8, 3, [0, 1, 2, 3, 4, 5, 6], [1, 2, 3]),
        (2, 3.1, [1, 2, 3, 4, 5, 6], [2, 3, 4]),
        (2.8, 3, [1, 2, 3, 4, 5, 6], [2, 3]),
        (1.8, 3.1, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4]),
        (2, 3, [-2, -1, 0, 1, 2, 3, 4], [2, 3]),
        (1.8, 3, [-2, -1, 0, 1, 2, 3, 4], [1, 2, 3]),
        (2, 3.1, [-2, -1, 0, 1, 2, 3, 4], [2, 3, 4]),
        (2.8, 3, [-2, -1, 0, 1, 2, 3, 4], [2, 3]),
        (1.8, 3.1, [-2, -1, 0, 1, 2, 3, 4], [1, 2, 3, 4]),
        (1.8, 4, [-2, -1, 0, 1, 2, 3, 4], [1, 2, 3, 4]),
        (2, 3, [6, 5, 4, 3, 2, 1, 0], [3, 2]),
        (1.8, 3, [6, 5, 4, 3, 2, 1, 0], [3, 2, 1]),
        (2, 3.1, [4, 3, 2, 1, 0, -1, -2], [4, 3, 2]),
    ),
)
def test_get_array_slice_indices(min_, max_, array, expected):
    array = np.array(array)
    expected = np.array(expected)
    min_idx, max_idx = raster.get_array_slice_indices(min_, max_, array)
    print(array[min_idx:max_idx])
    assert np.all(array[min_idx:max_idx] == expected)


@pytest.mark.parametrize(
    "array,lats,lons",
    (
        [[[1, 2, 3], [4, 5, 6]], list("xy"), list("abc")],
        [[[1, 2], [3, 4]], list("xy"), list("ab")],
        [[[1, 2], [3, 4], [5, 6]], list("xyz"), list("ab")],
    ),
)
def test_transform_array_to_df(array, lats, lons):
    assert len(array) == len(lats)
    assert len(array[0]) == len(lons)

    elements = []
    for row, lat in zip(array, lats):
        for value, lon in zip(row, lons):
            elements.append({"value": value, "lat": lat, "lon": lon})
    expected = pd.DataFrame(elements)

    df = raster.transform_array_to_df(array, lats, lons, geo=False)
    column_order = ["value", "lat", "lon"]
    assert expected[column_order].equals(df[column_order])
