"""
Utilities for working with GEE

- Aggregating rasters on GEE to polygons (FeatureCollections)
- Grouped aggregations
"""

# Standard
import os
import sys
import random
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Geospatial
import folium
import geopandas as gpd
import geemap

# Earth engine API
import ee

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Helper functions
def get_country_centroid(iso):
    """Read country boundaries and get a rough centroid"""
    import warnings

    gdf_centroid = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    gdf_centroid = gdf_centroid[gdf_centroid.iso_a3 == iso]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centroid_point = gdf_centroid.centroid.iloc[0]
    return [centroid_point.y, centroid_point.x]


# Process GEE
def load_admin_boundaries(cntry_selected_abbr, selected_admin_level, simplify_tol=100):
    """
    Load admin boundaries. Simplify country boundaries.
    """
    # Load administrative boundaries
    # country_boundaries_ee = ee.Feature(
    #     ee.FeatureCollection("FAO/GAUL/2015/level0")
    #     .filter(ee.Filter.eq("ADM0_NAME", cntry_selected))
    #     .first()
    # ).geometry()
    country_boundaries_ee = (
        ee.FeatureCollection(f"users/shreyasgm/growth_lab/gadm36_0")
        .filter(ee.Filter.eq("GID_0", cntry_selected_abbr))
        .first()
        .geometry()
        .simplify(maxError=simplify_tol)
    )
    admin_boundaries_ee = ee.FeatureCollection(
        f"users/shreyasgm/growth_lab/gadm36_{selected_admin_level}"
    ).filter(ee.Filter.eq("GID_0", cntry_selected_abbr))
    if selected_admin_level == 0:
        admin_boundaries_ee = admin_boundaries_ee.map(
            lambda x: x.simplify(maxError=simplify_tol)
        )
    return country_boundaries_ee, admin_boundaries_ee


def load_world_country_boundaries():
    return ee.FeatureCollection(f"users/shreyasgm/growth_lab/gadm36_0")


def load_city_boundaries(cntry_selected_abbr):
    city_boundaries_ee = ee.FeatureCollection(f"users/shreyasgm/growth_lab/ghs").filter(
        ee.Filter.eq("iso", cntry_selected_abbr)
    )
    return city_boundaries_ee


def get_custom_reducer():
    # Set custom reducer
    reducers = ee.Reducer.combine(
        ee.Reducer.combine(
            ee.Reducer.combine(
                ee.Reducer.combine(
                    ee.Reducer.combine(
                        ee.Reducer.count(), ee.Reducer.stdDev(), sharedInputs=True
                    ),
                    ee.Reducer.minMax(),
                    sharedInputs=True,
                ),
                ee.Reducer.median(),
                sharedInputs=True,
            ),
            ee.Reducer.mean(),
            sharedInputs=True,
        ),
        ee.Reducer.sum(),
        sharedInputs=True,
    )
    return reducers


def get_reduced_image(img, fc_boundaries, scaleFactor, set_date=False):
    """
    Reduce given image based on the following custom reducers:
    sum, mean, median, min, max, std dev, count
    """
    reducers = get_custom_reducer()
    reduced_img = img.reduceRegions(
        reducer=reducers,
        collection=fc_boundaries,
        scale=scaleFactor,
    )
    # Set date if necessary
    if set_date:
        img = img.set("system:time_start", ee.Number(set_date.timestamp() * 1000))
    # Add a date to each feature
    img_date = img.date().format()
    reduced_img = reduced_img.map(lambda x: x.set("date", img_date))
    return reduced_img


def get_reduced_image_grouped(
    img_with_bands,
    fc_boundaries,
    grouping_band_index,
    group_name,
    scaleFactor,
    set_date=False,
):
    """
    Reduce given image based on the following custom reducers:
    sum, mean, median, min, max, std dev, count
    """
    reducers = get_custom_reducer()
    reduced_img = img_with_bands.reduceRegions(
        reducer=reducers.group(groupField=grouping_band_index, groupName=group_name),
        collection=fc_boundaries,
        scale=scaleFactor,
    )
    # Set date if necessary
    if set_date:
        img = img.set("system:time_start", ee.Number(set_date.timestamp() * 1000))
    # Add a date to each feature
    img_date = img_with_bands.date().format()
    reduced_img = reduced_img.map(lambda x: x.set("date", img_date))
    return reduced_img


# function to get individual img dates
def get_date(img):
    return img.set("date", img.date().format())


def get_reduced_imagecollection(
    imagecollection,
    fc_boundaries,
    scaleFactor,
    grouped=False,
    grouping_band_index=1,
    group_name="group",
):
    # Get imagecollection reduced to featurecollection
    if not grouped:
        fx = lambda x: get_reduced_image(x, fc_boundaries, scaleFactor)
        reduced_fc = imagecollection.map(fx).flatten()
    else:
        fx = lambda x: get_reduced_image_grouped(
            x, fc_boundaries, grouping_band_index, group_name, scaleFactor
        )
        reduced_fc = imagecollection.map(fx).flatten()

    return reduced_fc


def extract_reduction_results(
    reduced_fc, id_cols, additional_stats=[], export_to_gcs=False, gcs_bucket=None
):
    """
    Extract reduction results into a dataframe

    Args:
        reduced_fc: FeatureCollection to export
        id_cols: ID columns as a list
        additional_stats: additional reducer stats if included in properties
        export_to_gcs: default False. Otherwise, string with CSV filename.
        gcs_bucket: str

    Returns:
        EE task if export_to_gcs. Otherwise, df with results.
    """
    # Convert data into EE lists
    stats_list = [
        "mean",
        "sum",
        "median",
        "stdDev",
        "count",
        "min",
        "max",
    ] + additional_stats
    id_cols = ["date"] + id_cols
    key_cols = id_cols + stats_list

    if export_to_gcs == False:
        areas_list = reduced_fc.reduceColumns(
            ee.Reducer.toList(len(key_cols)), key_cols
        ).values()
        # Force computation, convert to df
        results_df = pd.DataFrame(
            np.asarray(areas_list.getInfo()).squeeze(), columns=key_cols
        )
        # Convert date data type
        results_df["date"] = pd.to_datetime(results_df["date"])
        # Convert other data types
        for x in ["mean", "sum", "median", "stdDev", "min", "max"]:
            results_df[x] = results_df[x].astype(float)
        results_df["count"] = results_df["count"].astype(int)

        return results_df
    else:
        assert isinstance(export_to_gcs, str), "export_to_gcs should be a string"
        # Prepare and start export task
        export_task = ee.batch.Export.table.toCloudStorage(
            collection=reduced_fc,
            description=export_to_gcs,
            bucket=gcs_bucket,
            fileFormat="CSV",
            selectors=key_cols,
        )
        export_task.start()
        return export_task


def extract_reduction_results_grouped(reduced_fc, id_cols, grouping_cols):
    """
    Extract reduction results for grouped reductions
    """
    # Convert data into EE lists
    id_cols = ["date"] + id_cols
    key_cols = id_cols + ["groups"]

    areas_list = reduced_fc.reduceColumns(
        ee.Reducer.toList(len(key_cols)), key_cols
    ).values()
    # Force computation, convert to df
    results_df = pd.DataFrame(
        np.asarray(areas_list.getInfo(), dtype=object).squeeze(), columns=key_cols
    )
    # Explode list of dictionaries
    results_df = results_df.explode("groups")
    # Split dictionary elements
    results_df = pd.concat(
        [results_df[id_cols], results_df["groups"].apply(pd.Series)], axis=1
    )

    # Convert date data type
    results_df["date"] = pd.to_datetime(results_df["date"])
    # Convert other data types
    for x in ["mean", "sum", "median", "stdDev", "min", "max"]:
        results_df[x] = results_df[x].astype(float)
    results_df["count"] = results_df["count"].astype(int)
    return results_df
