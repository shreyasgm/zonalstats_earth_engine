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


def list_gcp_files(
    bucketname="earth_engine_aggregations", foldername=None, pattern=None
):
    """
    List files in GCP bucket / folder
    """
    from google.cloud import storage as gcs

    # List all files
    client = gcs.Client()
    blobs = client.list_blobs(bucketname, prefix=foldername)
    files = [x.name for x in blobs]
    # Check pattern match
    if pattern:
        r = re.compile(pattern)
        files = [x for x in files if r.match(x)]
    return files


def list_local_files(folderpath, pattern=None):
    """
    List local files in directory
    """
    # read filelist
    if pattern is None:
        pattern = "*.*"
    csvlist = list(Path(folderpath).glob(pattern))
    # Make sure file exists and has data in it
    csvlist_valid = [x for x in csvlist if x.stat().st_size > 10]
    return csvlist_valid


def download_missing_gcp(
    local_folderpath,
    gcs_bucketname="earth_engine_aggregations",
    gcs_foldername=None,
    pattern=None,
    n_jobs=None,
):
    from tqdm import tqdm
    from google.cloud import storage as gcs

    # List missing files
    local_files = list_local_files(folderpath=local_folderpath, pattern=pattern)
    local_files = [x.name for x in local_files]
    gcs_files = list_gcp_files(
        bucketname=gcs_bucketname, foldername=gcs_foldername, pattern=pattern
    )
    missing_files = list(set(gcs_files) - set(local_files))
    # Download
    client = gcs.Client()
    bucket = client.bucket(gcs_bucketname)

    def download_file_from_bucket(bucket, filename, local_folderpath):
        blob = bucket.blob(filename)
        blob.download_to_filename(Path(local_folderpath) / filename)

    if n_jobs is None:
        for filename in tqdm(missing_files):
            download_file_from_bucket(bucket, filename, local_folderpath)
        return missing_files
    else:
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs)(
            delayed(download_file_from_bucket)(bucket, f, local_folderpath)
            for f in tqdm(missing_files)
        )
        return missing_files
