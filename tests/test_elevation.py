import pytest
from unittest.mock import MagicMock

from projections.elevation import (
    map_lon_to_elevation_file,
    map_lat_to_elevation_file,
    map_shape_to_elevation_file,
)


@pytest.mark.parametrize(
    "lon,result",
    (
        (20, "w020"),
        (19.99, "w020"),
        (0, "w020"),
        (-19.99, "w020"),
        (-20, "w020"),
        (-21, "w060"),
        (21, "e020"),
        (-120, "w140"),
        (70, "e060"),
    ),
)
def test_map_lon_to_elevation_file(lon, result):
    assert map_lon_to_elevation_file(lon) == result


@pytest.mark.parametrize(
    "lat,result",
    (
        (0, "n40"),
        (-9.99, "n40"),
        (40, "n40"),
        (39.99, "n40"),
        (-10, "s10"),
        (41, "n90"),
    ),
)
def test_map_lat_to_elevation_file(lat, result):
    assert map_lat_to_elevation_file(lat) == result


@pytest.mark.parametrize(
    "bounds,results",
    (
        ((10, 0, 20, 10), ["gt30w020n40"]),
        ((-40, 80, -30, 90), ["gt30w060n90"]),
        ((-40, 35, -30, 45), ["gt30w060n40", "gt30w060n90"]),
        ((70, 36, 71, 37), ["gt30e060n40"]),
    ),
)
def test_map_lat_to_elevation_file(bounds, results):
    mock = MagicMock()
    mock.bounds = bounds
    files = map_shape_to_elevation_file(mock)
    for result in results:
        assert f"{result}.tif" in files
