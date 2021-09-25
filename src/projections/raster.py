import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry.polygon import Polygon
import xarray
from shapely.geometry import MultiPolygon, LineString


def create_by_separation(df, lat: str = "lat", lon: str = "lon"):
    sep = {}
    for col in [lat, lon]:
        uniques = np.sort(df[col].unique())
        diff = uniques[1:] - uniques[:-1]
        p05, p95 = np.percentile(diff, [5, 95])
        assert p05 == p95
        sep[col] = p05
    print("Separation:", sep)

    assert sep[lat] == sep[lon], "Only square buffers allowed"

    if not isinstance(df, gpd.GeoDataFrame):
        print("Converting to GeoDataFrame")
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]))

    df["raster"] = df.buffer(sep["lat"] / 2, cap_style=3)

    return df


def get_intersection_area(
    df,
    pol,
    value_col=None,
    lat="lat",
    lon="lon",
    raster="raster",
):
    if df.shape[0] == 0:
        return None

    row = gpd.GeoDataFrame(geometry=[pol], crs=4326)

    pol = row["geometry"].iloc[0]
    mask = slice(None)

    intersection = df.loc[mask, raster].intersection(pol).set_crs(epsg=4326)
    intersection_area = intersection.to_crs(epsg=3035).area

    if value_col:
        subdf = df.loc[mask, [lat, lon, value_col]].copy()
    else:
        subdf = df.loc[mask, [lat, lon]].copy()

    subdf["intersection_area"] = intersection_area

    return subdf


def weighted_pivot(df, weight="intersection_ratio", value_name="value", id_vars=None):
    # Probably not what you're looking for... See utils.get_weighted_average
    # The score of an adm2 is the average of all measures weighted by intersection_ratio
    value_cols = list(df.filter(regex=r"^\d\d-\d\d$").columns)
    df[value_cols] *= df[value_cols].multiply(df[weight], axis=0)

    if id_vars is None:
        id_vars = ["adm0", "adm1", "adm2"]

    aggfunc = {col: "mean" for col in value_cols}
    aggfunc[weight] = [np.sum, "count"]

    pivot = pd.pivot_table(
        df, index=id_vars, values=value_cols + [weight], aggfunc=aggfunc
    )

    pivot.columns = ["_".join(x) if x[0] == weight else x[0] for x in pivot.columns]

    pivot[value_cols] = pivot[value_cols].divide(pivot[f"{weight}_sum"], axis=0)

    # Reshape to long
    melt = pivot.reset_index().melt(
        id_vars=id_vars, value_vars=value_cols, var_name="time", value_name=value_name
    )

    melt = melt.merge(
        pivot[[f"{weight}_count", f"{weight}_sum"]], left_on=id_vars, right_index=True
    )

    return melt


def find_subset_with_intersection_area(array: xarray.DataArray, pol):
    subset = get_df_by_maximum_bounds(array, pol, geo=True)
    increment = get_increment_from_tif(array)
    include_square_raster(subset, increment=increment)
    subset = get_intersection_area(subset, pol, value_col="value", lat="lat", lon="lon")
    if subset is not None:
        subset = subset[subset["intersection_area"] > 0].copy()
    else:
        subset = pd.DataFrame()
    return subset


def get_df_by_maximum_bounds(array: xarray.DataArray, pol, geo=True):
    subarray, sublats, sublons = filter_array_by_maximum_bounds(array, pol)
    return transform_array_to_df(subarray, sublats, sublons, geo=geo)


def filter_array_by_maximum_bounds(array: xarray.DataArray, pol):
    lats = np.array(array.y)
    lons = np.array(array.x)

    min_lon, min_lat, max_lon, max_lat = pol.bounds

    min_lat_index, max_lat_index = get_array_slice_indices(min_lat, max_lat, lats)
    min_lon_index, max_lon_index = get_array_slice_indices(min_lon, max_lon, lons)
    lats = lats[min_lat_index:max_lat_index]
    lons = lons[min_lon_index:max_lon_index]
    arr = np.array(array[0, min_lat_index:max_lat_index, min_lon_index:max_lon_index])

    return arr, lats, lons


def get_array_slice_indices(min_value, max_value, array):
    min_index = find_index(min_value, array)
    max_index = find_index(max_value, array)

    max_element = array[max_index] if max_index < len(array) else array[-1]
    if max_element < max_value:
        max_index += 1

    max_index += 1  # Make the maximum inclusive

    return min_index, max_index


def find_index(value, array):
    if array[1] > array[0]:
        index = find_index_from_increasing_array(value, array)
    else:
        index = find_index_from_decreasing_array(value, array)
    return index


def find_index_from_increasing_array(value, array):
    increment = array[1] - array[0]
    min_ = array[0]
    value -= min_
    if value < 0:
        return 1

    idx = int(value // increment)
    if idx < 1:
        idx = 0

    return idx


def find_index_from_decreasing_array(value, array):
    increment = array[0] - array[1]
    min_ = array[-1]
    value -= min_
    if value < 0:
        return 0

    idx = int(value // increment)
    if idx < 1:
        idx = 1

    return len(array) - idx - 1


def reorder_indices(from_, to_):
    if from_ > to_:
        from_, to_ = to_, from_
    return from_, to_


def fill_no_data_value(array: np.array, no_data_value=None, fill_with=0):
    if no_data_value is None:
        no_data_mask = np.isnan(array)
    else:
        no_data_mask = array == no_data_value

    return np.where(no_data_mask, fill_with, array)


def transform_array_to_df(array, lats, lons, geo=True):
    obj = {
        "value": array.reshape(-1),
        "lon": np.repeat(lons, lats.shape[0]),
        "lat": np.tile(lats, lons.shape[0]),
    }
    if geo:
        df = gpd.GeoDataFrame(obj, geometry=gpd.points_from_xy(obj["lon"], obj["lat"]))
    else:
        df = pd.DataFrame(obj)
    return df


def get_increment_from_tif(tif):
    return abs(float(tif.x[0] - tif.x[1]))


def include_square_raster(df: gpd.GeoDataFrame, increment: float):
    df["raster"] = df.buffer(increment / 2, cap_style=3)


def include_rectangular_raster(
    df: gpd.GeoDataFrame, xincrement: float, yincrement: float
):
    def create_polygon(geo_pol):
        x, y, _, _ = geo_pol.bounds
        return Polygon(
            [
                (x - xincrement, y - yincrement),
                (x + xincrement, y - yincrement),
                (x + xincrement, y + yincrement),
                (x - xincrement, y + yincrement),
            ]
        )

    df["raster"] = df["geometry"].apply(create_polygon)


def filter_gdf_by_distance(df: gpd.GeoDataFrame, pol, max_distance):
    df["distance"] = df.distance(pol)
    return df[df["distance"] <= max_distance].copy()


def clip_raster(tif: xarray.DataArray, pol, crs):
    return tif.rio.clip([pol], crs)


def transform_xarray_to_gpd(array):
    obj = {
        "value": np.array(array).reshape(-1),
        "lon": np.repeat(array.x, array.shape[1]),
        "lat": np.tile(array.y, array.shape[2]),
    }
    df = gpd.GeoDataFrame(obj, geometry=gpd.points_from_xy(obj["lon"], obj["lat"]))
    return df


def quadrat_cut_geometry(geometry, quadrat_width: float, min_lines: int = 3):
    """
    Split a Polygon or MultiPolygon up into sub-polygons of a specified size.
    Parameters
    ----------
    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        the geometry to split up into smaller sub-polygons
    quadrat_width : numeric
        the linear width of the quadrats with which to cut up the geometry (in
        the units the geometry is in)
    min_lines : int
        the minimum number of linear quadrat lines (e.g., min_lines=3 would
        produce a quadrat grid of 4 squares)
    Returns
    -------
    geometry : shapely.geometry.MultiPolygon
    """
    # create n evenly spaced points between the min and max x and y bounds
    west, south, east, north = geometry.bounds
    x_num = int(np.ceil((east - west) / quadrat_width) + 1)
    y_num = int(np.ceil((north - south) / quadrat_width) + 1)
    x_points = np.linspace(west, east, num=max(x_num, min_lines))
    y_points = np.linspace(south, north, num=max(y_num, min_lines))

    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [
        LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points
    ]
    horizont_lines = [
        LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points
    ]
    lines = vertical_lines + horizont_lines

    # recursively split the geometry by each quadrat line
    for line in lines:
        geometry = MultiPolygon(shapely.ops.split(geometry, line))

    return geometry


def get_bounding_box_area(geometry) -> float:
    west, south, east, north = geometry.bounds
    return (north - south) * (east - west)


def is_shape_in_raster(shape, raster: xarray.DataArray):
    shape_min_lon, shape_min_lat, shape_max_lon, shape_max_lat = shape.bounds

    min_lat = min(float(raster.y[0]), float(raster.y[-1]))
    max_lat = max(float(raster.y[0]), float(raster.y[-1]))
    min_lon = min(float(raster.x[0]), float(raster.x[-1]))
    max_lon = max(float(raster.x[0]), float(raster.x[-1]))

    lat_belongs = (min_lat <= shape_max_lat <= max_lat) or (
        min_lat <= shape_min_lat <= max_lat
    )
    lon_belongs = (min_lon <= shape_min_lon <= max_lon) or (
        min_lon <= shape_max_lon <= max_lon
    )

    return lat_belongs and lon_belongs


def merge_df_to_array_by_lat_lon(df: pd.DataFrame, array: xarray.DataArray, pol):
    df = df.copy()
    subarray = get_df_by_maximum_bounds(array, pol, geo=False)

    df["lat"] = np.round(df["lat"], 6)
    df["lon"] = np.round(df["lon"], 6)
    subarray["lat"] = np.round(subarray["lat"], 6)
    subarray["lon"] = np.round(subarray["lon"], 6)

    subdf = df.drop(columns=["value"]).merge(subarray, on=["lat", "lon"])
    return subdf
