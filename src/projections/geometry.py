import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

from projections.constants import FLAT_CRS


def contains(lat, lon, shape):
    point = Point(float(lon), float(lat))
    return point.within(shape)


def find_record(lat, lon, records):
    for record in records:
        shape = record[0]
        if contains(lat, lon, shape):
            return record


def nearest_loc_match(adm, locs, columns, verbose=True):
    mask = locs["index_left"].isnull()

    # To Flat CRS
    adm.to_crs(FLAT_CRS, inplace=True)
    locs.to_crs(FLAT_CRS, inplace=True)

    if verbose:
        iterator = tqdm(locs[mask].iterrows(), total=mask.sum(), desc="Finding codes")
    else:
        iterator = locs[mask].iterrows()

    locs["nearest_loc"] = False
    for idx, row in iterator:
        argmin = adm.distance(row["geometry"]).argmin()
        match = adm.iloc[argmin]

        for col in columns:
            locs.loc[idx, col] = match[col]
        locs.loc[idx, "nearest_loc"] = True

    return locs


def loc_match(adm, locs, columns_of_interest, verbose=True):
    locs = gpd.sjoin(adm, locs, how="right", op="contains")

    if verbose:
        print("Exact matches:", locs["index_left"].notnull().sum())

    locs = nearest_loc_match(adm, locs, columns_of_interest, verbose=verbose)
    return locs
