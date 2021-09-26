import pytest
import numpy as np
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
