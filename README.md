# Projections

[projections](src/projections) is a collection of utils written to combine geographical data in various formats (e.g. [TIF](src/projections/utils.py), [NC](src/projections/temperature.py), [TXT](src/projections/elevation.py), [HDF](src/projections/modis.py)) into a set of agregations in tabular format. These modules are tailored for the data they were written for, thus the naming and organisation of the files makes reference to these specific usecases, so it may be necessary to adapt some routines to acommodate new sources of data.

Specific applications of these methods can be found in the [Notebooks](src/projections/Notebooks) directory. As an example, the main elements of [GPCP](https://psl.noaa.gov/data/gridded/data.gpcp.html) processing are highligted below (see [full notebook](src/projections/Notebooks/GPCP.ipynb) for more details):

```python
# Read data
converter = NcConverter(nodata_name="missing_value")
converter.read(read_path / filename)
IMAGE = converter.get_xarray("precip", period=0, lon_offset=-180)

geo_df = gpd.read_file('../Shapefiles/preprocessed/all_countries_with_eth.shp')

# Process all geometries
iterator = partial(utils.yield_missing_shapes, save_path=partial_path, prefix='p0')
n_processes = 30
with ProcessPoolExecutor(n_processes) as ppe:
    for _ in tqdm(
        ppe.map(save_location_mapping, iterator(geo_df)),
        total=geo_df.shape[0]
    ):
        pass

# Aggregate and save into a single output file
def aggregate(image):
    agg = Aggregator(
        by_country_path=by_country_path,
        partial_path=partial_path,
        in_memory=True,
        mapping_dfs=mapping_dfs
    )
    agg.aggregate(image)

for variable in ("precip",):
    with ProcessPoolExecutor(n_processes) as tpe:
        for name in tpe.map(aggregate,
                            converter.iter_periods(attribute_name=variable, lon_offset=-180)):
            pass
```

# Datasets

| Name        | Coordinates                                                   | Raster                                                               | Notes                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------- | ------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CRU         | [Yes](src/projections/Notebooks/Precipitaciones_cru.ipynb)    | [Yes](src/projections/Notebooks/Precipitaciones_cru_raster.ipynb)    | There are 4 output variables representing different aggregations of precipitation: count, mean, median and std. There are 13 output files, each with a different aggregation level. There are yearly and monthly files, as well a 3 geographic levels: country (ADM0), edo (ADM1) and mun (ADM0). Those with `nnl` in their names exclude inexact matches (coordinate-based). `cru.csv` has disaggregated data.   |
| Elevation   | No                                                            | [Yes](src/projections/Notebooks/Elevation.ipynb)                     | The output variable is `value`, which represents a weighted by intersection area of all the grids that intersected the underlying polygon.                                                                                                                                                                                                                                                                        |
| FAO         | No                                                            | [Yes](src/projections/Notebooks/AgroAllFAO.ipynb)                    | The output is a wide dataset with 449 columns. Apart from the `id` and the `intersection_area`, each colup represent a crop and type of measurement following the format `Agro-[TypeOfMeasurement]_[Name]`. For example, `Agro-ClimaticPotentialYield_Alfalfa` comes from measuring `ClimaticPotentialYield_Alfalfa` for Alfalfa.                                                                                 |
| GDELT       | [Yes](src/projections/Notebooks/GDELT.ipynb)                  | No                                                                   | Data from the GDELT project. It has many output variables: EventCode, NumMentions, IsRootEvent, NumSources, NumArticles and AvgTone. See the [user manual](http://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf) for further reference. Each variable is aggregated by calculating their sum, mean, median and std.                                                                          |
| GPCC        | [Yes](src/projections/Notebooks/Precipitaciones_GPCC.ipynb)   | [Yes](src/projections/Notebooks/Precipitaciones_GPCC_raster.ipynb)   | There are 4 output variables representing different aggregations of precipitation: count, mean, median and std. There are 13 output files, each with a different aggregation level. There are yearly and monthly files, as well a 3 geographic levels: country (ADM0), edo (ADM1) and mun (ADM0). Those with `nnl` in their names exclude inexact matches (coordinate-based). `GPCC.csv` has disaggregated data.  |
| GPCP        | No                                                            | [Yes](src/projections/Notebooks/GPCP.ipynb)                          | Both sums and weighted averages are provided per year and type. Variables are annotated with the pattern `[average\|sum]_[variable]_[year]`. In addition, n_grids counts the number of grids from the underlying TIF file that were aggregated in order to obtain the result.                                                                                                                                     |
| Nighlights  | [Yes](src/projections/Notebooks/NL_Subnat.ipynb)              | [Yes](src/projections/Notebooks/NL_NOAA.ipynb)                       |                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Population  | No                                                            | [Yes](src/projections/Notebooks/Population.ipynb)                    | This comprises two sets of information: population count and density. To avoid loosing detail when aggregation, both sums and weighted averages are provided per year and type. Variables are annotated with the pattern `[average\|sum]\_[Count\|Density]\_[year]`. In addition, n_grids counts the number of grids from the underlying TIF file that were aggregated in order to obtain the result.             |
| Ruggedness  | No                                                            | [Yes](src/projections/Notebooks/Elevation-Ruggedness.ipynb)          | There are 3 output variables, each representing a different input: cellarea, slope and tri. All are a weighted by intersection area of all the grids that intersected the underlying polygon.                                                                                                                                                                                                                     |
| SCAD        | No                                                            | [Yes](src/projections/Notebooks/SCAD.ipynb)                          | There are 16 output files, each with a different aggregation level. There are yearly, monthly and weekly files, as well a 4 geographic levels: country (ADM0), edo (ADM1), mun (ADM0) and ethnic (Africa only). `SCAD.csv` has disaggregated data.                                                                                                                                                                |
| SCPDSI      | [Yes](src/projections/Notebooks/Precipitaciones_scpdsi.ipynb) | [Yes](src/projections/Notebooks/Precipitaciones_scpdsi_raster.ipynb) | here are 4 output variables representing different aggregations of precipitation: count, mean, median and std. There are 13 output files, each with a different aggregation level. There are yearly and monthly files, as well a 3 geographic levels: country (ADM0), edo (ADM1) and mun (ADM0). Those with `nnl` in their names exclude inexact matches (coordinate-based). `scpdsi.csv` has disaggregated data. |
| Temperature | No                                                            | [Yes](src/projections/Notebooks/Temperature.ipynb)                   |                                                                                                                                                                                                                                                                                                                                                                                                                   |

# Aggregating results

Results of aggregating rasters can be aggregated further in order to work with bigger polygons. For example, the value for `USA.39_1` will be the average of all values of `USA.39.*_1` weighted by their `intersection_area`. In the case of Population count, as this is measured with a sum, in order to aggregate simply sum the results of each internal polygon.

```Python
# Aggregate averaged value
df["weighted_value"] = df["intersection_area"] * df["value"]
df.groupby("id")["weighted_value"].sum() / df.groupby("id")["intersection_area"].sum()

# Aggregate summed value
df.groupby("id")["value"].sum()
```
