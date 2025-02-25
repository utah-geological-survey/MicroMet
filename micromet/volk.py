import logging
import numbers
from typing import Optional, Tuple, Union

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import configparser
import pandas as pd
import numpy as np
import pathlib
import pyproj
import rasterio
import logging
import multiprocessing as mp
import datetime
from affine import Affine
from fluxdataqaqc import Data
from matplotlib.colors import LogNorm
from numpy import ma
from scipy import signal as sg
from scipy.ndimage import gaussian_filter

import requests
from pathlib import Path

import xarray
import refet

###############################################################################
# Configure logging
###############################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# You can send logs to stdout, a file, or elsewhere. Here we just use StreamHandler:
stream_handler = logging.StreamHandler()
# Customize the log format
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s\n"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# Define the file path (absolute or relative). For instance:
log_file_path = "../logs/volk.log"

# Create a FileHandler and set the level
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.WARNING)

# Create a Formatter
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s\n"
)

# Set the formatter for the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

###############################################################################


def _load_configs(station,
                  config_path='../../station_config/',
                  secrets_path="../../secrets/config.ini"):
    """
    Load station metadata and secrets from configuration files.

    Parameters:
    ----------
    station : str
        Station identifier.
    config_path : str
        Path to station configuration file.
    secrets_path : str
        Path to secrets configuration file.

    Returns:
    -------
    dict
        A dictionary containing station metadata and database URL.
    """

    if isinstance(config_path, Path):
        pass
    else:
        config_path = Path(config_path)

    config_path_loc = config_path / f'{station}.ini'
    config = configparser.ConfigParser()
    config.read(config_path_loc)

    if isinstance(secrets_path, Path):
        pass
    else:
        secrets_path = Path(secrets_path)

    secrets_config = configparser.ConfigParser()
    secrets_config.read(secrets_path)

    return {
        "url": secrets_config["DEFAULT"]["url"],
        "latitude": float(config["METADATA"]["station_latitude"]),
        "longitude": float(config["METADATA"]["station_longitude"]),
        "elevation": float(config["METADATA"]["station_elevation"])
    }


def _fetch_and_preprocess_data(url, station, startdate):
    """
    Retrieve flux data and preprocess it.

    Parameters:
    ----------
    url : str
        Database URL.
    station : str
        Station identifier.
    startdate : str
        Start date for data retrieval.

    Returns:
    -------
    pd.DataFrame
        Preprocessed dataframe of flux data.
    """
    headers = {'Accept-Profile': 'groundwater', 'Content-Type': 'application/json'}
    params = {'stationid': f'eq.{station}', 'datetime_start': f'gte.{startdate}'}

    try:
        response = requests.get(f"{url}/amfluxeddy", headers=headers, params=params)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df['datetime_start'] = pd.to_datetime(df['datetime_start'])
        df = df.set_index('datetime_start')
        df.replace(-9999, np.nan, inplace=True)
        df = df.resample('1h').mean(numeric_only=True)
        df.dropna(subset=['h2o', 'wd', 'ustar', 'v_sigma'], inplace=True)
        return df
    except requests.RequestException as e:
        logging.error(f"Failed to fetch data for station {station}: {e}")
        return pd.DataFrame()


def _compute_hourly_footprint(temp_df, station_x, station_y, zm, h_s, z0, dx, origin_d):
    """
    Compute hourly footprint climatology.

    Parameters:
    ----------
    temp_df : pd.DataFrame
        Filtered dataframe for a specific day and time window.
    station_x, station_y : float
        UTM coordinates of the station.
    zm : float
        Adjusted measurement height.
    h_s : float
        Atmospheric boundary layer height.
    z0 : float
        Roughness length.
    dx : float
        Model resolution.
    origin_d : float
        Model domain boundary.

    Returns:
    -------
    list
        List of tuples containing hour, footprint array, and metadata.
    """
    footprints = []
    for hour in range(6, 19):  # From 7 AM to 8 PM
        temp_line = temp_df[temp_df.index.hour == hour]
        if temp_line.empty:
            logging.warning(f"No data for {hour}:00, skipping.")
            continue

        try:
            ffp_result = ffp_climatology(
                domain=[-origin_d, origin_d, -origin_d, origin_d],
                dx=dx, dy=dx, zm=zm, h=h_s, rs=None, z0=z0,
                ol=temp_line['mo_length'].values,
                sigmav=temp_line['v_sigma'].values,
                ustar=temp_line['ustar'].values,
                umean=temp_line['ws'].values,
                wind_dir=temp_line['wd'].values,
                crop=0, fig=0, verbosity=0
            )

            f_2d = np.array(ffp_result['fclim_2d']) * dx ** 2
            x_2d = np.array(ffp_result['x_2d']) + station_x
            y_2d = np.array(ffp_result['y_2d']) + station_y
            f_2d = mask_fp_cutoff(f_2d)

            footprints.append((hour, f_2d, x_2d, y_2d))
        except Exception as e:
            logging.error(f"Error computing footprint for hour {hour}: {e}")
            continue

    return footprints

def _write_footprint_to_raster(footprints, output_path, epsg=5070):
    """
    Write computed footprint climatology to a GeoTIFF raster file.

    Parameters:
    ----------
    footprints : list of tuples
        Each tuple contains (hour, footprint array, x_2d, y_2d).
    output_path : pathlib.Path
        Path to the output raster file.
    """
    if not footprints:
        logging.warning(f"No footprints to write for {output_path}. Skipping.")
        return

    try:
        first_footprint = footprints[0][1]
        transform = find_transform(footprints[0][2], footprints[0][3])
        n_bands = len(footprints)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            dtype=rasterio.float64,
            count=n_bands,
            height=first_footprint.shape[0],
            width=first_footprint.shape[1],
            transform=transform,
            crs=epsg,  # Ensure this matches the projection used in `pyproj`
            nodata=0.0
        ) as raster:

            for i, (hour, f_2d, _, _) in enumerate(footprints, start=1):
                raster.write(f_2d, i)
                raster.update_tags(i, hour=f'{hour:02}00', total_footprint=np.nansum(f_2d))

        logging.info(f"Footprint raster saved: {output_path}")

    except Exception as e:
        logging.error(f"Failed to write raster {output_path}: {e}")

def weighted_rasters(
    stationid='US-UTW',
    start_hr=6,
    end_hr=18,
    normed_NLDAS_stats_path='nldas_all_normed.parquet',
    out_dir=None
):
    """
    Generate daily weighted footprint rasters based on hourly fetch and normalized ETo data.

    This function reads a Parquet file containing daily normalized ETo values for multiple
    stations, filters it by the specified station ID, and applies hourly weighting to existing
    footprint rasters. For each unweighted TIFF file in `out_dir`, the function:
    1. Parses the date from the filename.
    2. Reads hourly bands from the raster, normalizes them by their global sum, and multiplies
       by the normalized ETo value for that hour.
    3. Sums these hourly weighted rasters into a single daily footprint raster.
    4. Writes the result to a new file named `<YYYY-MM-DD>_weighted.tif` in `out_dir`.

    Parameters
    ----------
    stationid : str, optional
        Station identifier used to look up normalized ETo values (default: 'US-UTW').
    start_hr : int, optional
        Starting hour for the data slice (default: 6, i.e. 7 AM).
    end_hr : int, optional
        Ending hour for the data slice (default: 18, i.e. 8 PM).
    normed_NLDAS_stats_path : str or pathlib.Path, optional
        Path to the Parquet file containing normalized ETo data (default: 'nldas_all_normed.parquet').
    out_dir : str or pathlib.Path, optional
        Output directory containing unweighted footprint rasters (default: current directory).

    Notes
    -----
    - Any TIFF files in `out_dir` with filenames starting with '20' (e.g., '2022-01-01.tif') are
      processed, unless they already contain the substring 'weighted' in their filename.
    - The function expects the TIFF filename to be in the form 'YYYY-MM-DD.tif' so it can parse
      out the date.
    - Only generates a weighted TIFF file if the total sum of the final footprint is within 0.15
      of 1.0.
    - Hourly rasters with all NaN values are replaced with zeros.
    - Written rasters preserve the same georeferencing, resolution, and coordinate reference
      system as the input rasters.

    Returns
    -------
    None
        This function does not return anything. It writes a single-band, daily-weighted footprint
        raster to `out_dir` for each processed date.

    Example
    -------
    >>> weighted_rasters(
    ...     stationid='US-UTW',
    ...     start_hr=6,
    ...     end_hr=18,
    ...     normed_NLDAS_stats_path='nldas_all_normed.parquet',
    ...     out_dir='/path/to/tif/files'
    ... )
    """
    # Ensure out_dir is a Path
    if out_dir is None:
        out_dir = pathlib.Path('./output/')
    else:
        out_dir = pathlib.Path(out_dir)

    # Read the Parquet file and filter data for the specified station
    eto_df = pd.read_parquet(normed_NLDAS_stats_path)
    eto_df['daily_ETo_normed'] = eto_df['daily_ETo_normed'].fillna(0)  # Fill missing ETo with 0
    nldas_df = eto_df.loc[stationid]

    # Iterate over all TIFF files in out_dir that begin with '20'
    for out_f in out_dir.glob('20*.tif'):
        # Skip if the file is already weighted
        if 'weighted' in out_f.stem:
            continue

        logging.info(f"Processing {out_f.name}")

        # Parse the date from the file name
        try:
            date = datetime.datetime.strptime(out_f.stem, '%Y-%m-%d')
        except ValueError:
            logging.warning(f"Skipping {out_f.name} because its filename is not in 'YYYY-MM-DD' format.")
            continue

        # Prepare output file name
        final_outf = out_dir / f"{date.year:04d}-{date.month:02d}-{date.day:02d}_weighted.tif"

        # Skip if output already exists
        if final_outf.is_file():
            logging.info(f"Weighted file already exists for {date.date()}. Skipping.")
            continue

        # Open the source raster once
        with rasterio.open(out_f) as src:
            band_indexes = src.indexes  # e.g. [1, 2, 3, ...] for each hour
            # We'll accumulate the weighted footprint across all hours
            normed_fetch_rasters = []

            for band_idx in band_indexes:
                # The hour we are processing: band 1 corresponds to (start_hr), band 2 -> (start_hr+1), etc.
                hour = band_idx + start_hr - 1
                dtindex = pd.to_datetime(f"{date:%Y-%m-%d} {hour:02d}:00:00")

                # Attempt to read the normalized ETo from nldas_df
                try:
                    norm_eto = nldas_df.loc[dtindex, 'daily_ETo_normed']
                except KeyError:
                    logging.warning(f"No NLDAS record for {dtindex}; using 0 as fallback.")
                    norm_eto = 0.0

                arr = src.read(band_idx)
                band_sum = np.nansum(arr)

                # Avoid division by zero
                if band_sum == 0 or np.isnan(band_sum):
                    # If everything is NaN or zero, use a zeros array
                    tmp = np.zeros_like(arr)
                else:
                    # Normalize by band sum
                    tmp = arr / band_sum

                # Multiply by normalized ETo
                weighted_arr = tmp * norm_eto
                normed_fetch_rasters.append(weighted_arr)

            # Sum the weighted hourly rasters into a single daily footprint
            final_footprint = sum(normed_fetch_rasters)

            # Only proceed if the daily sum is close to 1.0
            footprint_sum = final_footprint.sum()
            if np.isclose(footprint_sum, 1.0, atol=0.15):
                # Write output raster
                logging.info(f"Writing weighted footprint to {final_outf}")
                with rasterio.open(
                    final_outf, 'w',
                    driver='GTiff',
                    dtype=rasterio.float64,
                    count=1,
                    height=final_footprint.shape[0],
                    width=final_footprint.shape[1],
                    transform=src.transform,
                    crs=src.crs,
                    nodata=0.0
                ) as out_raster:
                    out_raster.write(final_footprint, 1)
            else:
                logging.warning(
                    f"Final footprint sum check failed for {date.date()}: sum={footprint_sum:.3f}"
                )

def clip_to_utah_merge(file_dir="./NLDAS_data/", years=None):
    """
    Clip NLDAS NetCDF files to Utah boundaries and merge them by year.

    This function scans a specified directory for NetCDF files matching the given
    `years`, extracts data within Utah's latitude and longitude bounds, and merges
    them along the time dimension. The final merged dataset is saved both in NetCDF
    and Parquet formats for easy downstream use. If no `years` are specified, defaults
    to [2022, 2023, 2024].

    Parameters
    ----------
    file_dir : str or pathlib.Path, optional
        Directory path containing NLDAS NetCDF files (default: "./NLDAS_data/").
    years : list of int, optional
        List of years for which files will be processed (default: [2022, 2023, 2024]).

    Notes
    -----
    - The latitude and longitude bounds for Utah are hard-coded:
        * lat: 37.0 to 42.0
        * lon: -114.0 to -109.0
    - The function merges all files from a given year along the `time` dimension using
      `xarray.concat`.
    - Each merged dataset is saved to:
        * NetCDF:  `<year>_utah_merged.nc`
        * Parquet: `<year>_utah_merged.parquet`
    - The function prints progress information and the filenames of created outputs.

    Returns
    -------
    None
        This function does not return anything. It saves output files to the working directory.

    Example
    -------
    >>> clip_to_utah_merge(file_dir="./NLDAS_data/", years=[2021, 2022])
    """
    # Define Utah's latitude and longitude boundaries
    utah_lat_min, utah_lat_max = 37.0, 42.0
    utah_lon_min, utah_lon_max = -114.0, -109.0

    if isinstance(file_dir, Path):
        netcdf_files = file_dir
    else:
        # List of uploaded NetCDF files
        netcdf_files = pathlib.Path(file_dir)

    if isinstance(years, list):
        pass
    else:
        years = [2022, 2023, 2024]

    for year in years:
        print(year)
        # Extract Utah-specific data from each file and store datasets
        utah_datasets = []
        for file in netcdf_files.glob(f"{year}*.nc"):
            print(file)
            ds_temp = xarray.open_dataset(file)
            ds_utah_temp = ds_temp.sel(
                lat=slice(utah_lat_min, utah_lat_max),
                lon=slice(utah_lon_min, utah_lon_max),
            )
            utah_datasets.append(ds_utah_temp)

        # Merge all extracted datasets along the time dimension
        ds_merged = xarray.concat(utah_datasets, dim="time")

        # Save as NetCDF using a compatible format (default for xarray in this environment)
        netcdf_output_path = f"{year}_utah_merged.nc"
        ds_merged.to_netcdf(netcdf_output_path)

        # Convert to Pandas DataFrame for Parquet format
        df_parquet = ds_merged.to_dataframe().reset_index()

        # Save as Parquet
        parquet_output_path = f"{year}_utah_merged.parquet"
        df_parquet.to_parquet(parquet_output_path, engine="pyarrow")

        # Provide download links
        print(netcdf_output_path, parquet_output_path)

def calc_nldas_refet(date, hour, nldas_out_dir, latitude, longitude, elevation, zm):
    """
    Calculate reference evapotranspiration (ETr and ETo) using NLDAS data for a specific
    date, hour, and point location, then append or create a CSV time series of results.

    This function:
    1. Constructs a file path based on the specified year, month, day, and hour.
    2. Opens the corresponding NLDAS GRIB file using `xarray` and extracts the nearest grid
       cell to the given latitude and longitude.
    3. Computes hourly vapor pressure, wind speed, temperature, and solar radiation from
       the dataset.
    4. Uses the `refet` library to calculate hourly reference evapotranspiration (ETr) and
       reference evaporation (ETo) using the ASCE method.
    5. Creates or updates a CSV file (`nldas_ETr.csv`) with the calculated ETr/ETo values
       and relevant meteorological variables for the specified datetime.
    6. Returns the updated DataFrame containing all ETr/ETo records up to the current datetime.

    Parameters
    ----------
    date : datetime.datetime
        The date for which to calculate reference ET.
    hour : int
        The hour (0-23) for which to calculate reference ET.
    nldas_out_dir : str or pathlib.Path
        Directory containing hour-specific NLDAS GRIB files (e.g., "YYYY_MM_DD_HH.grb").
    latitude : float
        The latitude of the point of interest.
    longitude : float
        The longitude of the point of interest.
    elevation : float
        The elevation (in meters) of the point of interest.
    zm : float
        Measurement (wind) height above the ground, in meters.

    Returns
    -------
    pandas.DataFrame
        A DataFrame (indexed by datetime) containing the updated ETr, ETo, and related
        meteorological variables (vapor pressure, specific humidity, wind speed, air
        pressure, temperature, solar radiation).

    Notes
    -----
    - The function uses the `pynio` engine for reading GRIB files with `xarray`.
    - Vapor pressure, wind speed, temperature, and solar radiation are computed from the
      NLDAS variables:
        * `PRES_110_SFC` (air pressure in Pa),
        * `SPF_H_110_HTGL` (specific humidity in kg/kg),
        * `U_GRD_110_HTGL` / `V_GRD_110_HTGL` (wind components in m/s),
        * `TMP_110_HTGL` (air temperature in K),
        * `DSWRF_110_SFC` (downward shortwave radiation in W/m²).
    - It expects a valid NLDAS GRIB file matching the pattern "YYYY_MM_DD_HH.grb" located
      in `nldas_out_dir`. Otherwise, an error may occur.
    - The function writes results to a CSV file named `nldas_ETr.csv` within the directory
      `All_output/AMF/<station>` (the `station` variable is assumed to be defined elsewhere).

    Example
    -------
    >>> from datetime import datetime
    >>> calc_nldas_refet(
    ...     date=datetime(2023, 7, 15),
    ...     hour=12,
    ...     nldas_out_dir=Path("./NLDAS_data"),
    ...     latitude=40.0,
    ...     longitude=-111.9,
    ...     elevation=1500,
    ...     zm=2.0
    ... )
    """
    YYYY = date.year
    DOY = date.timetuple().tm_yday
    MM = date.month
    DD = date.day
    HH = hour
    # already ensured to exist above loop
    nldas_outf_path = nldas_out_dir / f"{YYYY}_{MM:02}_{DD:02}_{HH:02}.grb"
    # open grib and extract needed data at nearest gridcell, calc ETr/ETo anf append to time series
    ds = xarray.open_dataset(nldas_outf_path, engine="pynio").sel(
        lat_110=latitude, lon_110=longitude, method="nearest"
    )
    # calculate hourly ea from specific humidity
    pair = ds.get("PRES_110_SFC").data / 1000  # nldas air pres in Pa convert to kPa
    sph = ds.get("SPF_H_110_HTGL").data  # kg/kg
    ea = refet.calcs._actual_vapor_pressure(q=sph, pair=pair)  # ea in kPa
    # calculate hourly wind
    wind_u = ds.get("U_GRD_110_HTGL").data
    wind_v = ds.get("V_GRD_110_HTGL").data
    wind = np.sqrt(wind_u**2 + wind_v**2)
    # get temp convert to C
    temp = ds.get("TMP_110_HTGL").data - 273.15
    # get rs
    rs = ds.get("DSWRF_110_SFC").data
    unit_dict = {"rs": "w/m2"}
    # create refet object for calculating

    refet_obj = refet.Hourly(
        tmean=temp,
        ea=ea,
        rs=rs,
        uz=wind,
        zw=zm,
        elev=elevation,
        lat=latitude,
        lon=longitude,
        doy=DOY,
        time=HH,
        method="asce",
        input_units=unit_dict,
    )  # HH must be int

    out_dir = Path("All_output") / "AMF" / f"{station}"

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    # this one is saved under the site_ID subdir
    nldas_ts_outf = out_dir / f"nldas_ETr.csv"
    # save/append time series of point data
    dt = pd.datetime(YYYY, MM, DD, HH)
    ETr_df = pd.DataFrame(
        columns=["ETr", "ETo", "ea", "sph", "wind", "pair", "temp", "rs"]
    )
    ETr_df.loc[dt, "ETr"] = refet_obj.etr()[0]
    ETr_df.loc[dt, "ETo"] = refet_obj.eto()[0]
    ETr_df.loc[dt, "ea"] = ea[0]
    ETr_df.loc[dt, "sph"] = sph
    ETr_df.loc[dt, "wind"] = wind
    ETr_df.loc[dt, "pair"] = pair
    ETr_df.loc[dt, "temp"] = temp
    ETr_df.loc[dt, "rs"] = rs
    ETr_df.index.name = "date"

    # if first run save file with individual datetime (hour data) else open and overwrite hour
    if not nldas_ts_outf.is_file():
        ETr_df.round(4).to_csv(nldas_ts_outf)
        nldas_df = ETr_df.round(4)
    else:
        curr_df = pd.read_csv(nldas_ts_outf, index_col="date", parse_dates=True)
        curr_df.loc[dt] = ETr_df.loc[dt]
        curr_df.round(4).to_csv(nldas_ts_outf)
        nldas_df = curr_df.round(4)

    return nldas_df

def calc_hourly_ffp_xr(input_data_dir=None, years=None, output_dir=None, ):
    if years is None:
        years = [2021,2022,2023,2024]
    else:
        years = years

    if input_data_dir is None:
        input_data_dir = Path("./output/")
    elif isinstance(input_data_dir, Path):
        input_data_dir = input_data_dir
    else:
        input_data_dir = Path(input_data_dir)

    if output_dir is None:
        output_dir = input_data_dir
    elif isinstance(output_dir, Path):
        output_dir = output_dir
    else:
        output_dir = Path(output_dir)

    for year in years:
        print(year)

        ds = xarray.open_dataset(input_data_dir / f"{year}_utah_merged.nc", )

        # Convert temperature to Celsius
        temp = ds["Tair"].values - 273.15

        # Compute actual vapor pressure (ea)
        pair = ds["PSurf"].values / 1000  # Convert pressure from Pa to kPa
        sph = ds["Qair"].values  # Specific humidity (kg/kg)
        ea = refet.calcs._actual_vapor_pressure(q=sph, pair=pair)  # Vapor pressure (kPa)

        # Compute wind speed from u and v components
        wind_u = ds["Wind_E"].values
        wind_v = ds["Wind_N"].values
        uz = np.sqrt(wind_u ** 2 + wind_v ** 2)  # Wind speed (m/s)

        # Extract shortwave radiation
        rs = ds["SWdown"].values  # Solar radiation (W/m²)

        # Extract time variables
        time_vals = ds["time"].values  # Convert to numpy datetime64
        dt_index = pd.to_datetime(time_vals)  # Convert to Pandas datetime index
        DOY = dt_index.dayofyear.values  # Day of year
        HH = dt_index.hour.values  # Hour of day
        # Expand DOY and HH to match (time, lat, lon) shape
        doy_expanded = np.broadcast_to(DOY[:, np.newaxis, np.newaxis], temp.shape)
        hh_expanded = np.broadcast_to(HH[:, np.newaxis, np.newaxis], temp.shape)

        # Define measurement height (assumed)
        zw = 2.0  # Wind measurement height in meters

        # Define elevation range (664m to 4125m, step 100m)
        elevation_range = np.arange(1100, 2000, 25)

        # Create an empty array to store ETo values
        eto_results = np.zeros((len(elevation_range),) + temp.shape)  # Shape (elevations, time, lat, lon)
        etr_results = np.zeros((len(elevation_range),) + temp.shape)

        # Loop over elevations and compute ETo
        for i, elev in enumerate(elevation_range):
            refet_obj = refet.Hourly(
                tmean=temp, ea=ea, rs=rs, uz=uz,
                zw=2, elev=elev, lat=ds["lat"].values, lon=ds["lon"].values,
                doy=doy_expanded, time=hh_expanded, method="asce", input_units={"rs": "w/m2"}
            )
            eto_results[i] = refet_obj.eto()  # Store ETo results for each elevation
            etr_results[i] = refet_obj.etr()  # Store ETr results for each elevation

        # Convert ETo results to an xarray DataArray
        eto_da = xarray.DataArray(
            data=eto_results,
            dims=("elevation", "time", "lat", "lon"),
            coords={
                "elevation": elevation_range,
                "time": ds["time"],
                "lat": ds["lat"],
                "lon": ds["lon"]
            },
            attrs={"units": "mm/hour",
                   "description": "Hourly reference evapotranspiration (ASCE) at different elevations"}
        )

        # Convert ETo results to an xarray DataArray
        etr_da = xarray.DataArray(
            data=etr_results,
            dims=("elevation", "time", "lat", "lon"),
            coords={
                "elevation": elevation_range,
                "time": ds["time"],
                "lat": ds["lat"],
                "lon": ds["lon"]
            },
            attrs={"units": "mm/hour",
                   "description": "Hourly reference evapotranspiration (ASCE) at different elevations"}
        )

        # Add ETo to the dataset
        ds = ds.assign(ETo=eto_da)
        # Add ETo to the dataset
        ds = ds.assign(ETr=etr_da)

        # Save the modified dataset (Optional)
        ds.to_netcdf(output_dir / f"{year}_with_eto.nc")


def calc_hourly_ffp(station, startdate='2022-01-01', config_path=None, secrets_path='../../secrets/config.ini',
             epsg=5070, h_c=0.2, zm_s=2.0, dx=3.0, h_s=2000.0, origin_d=200.0):
    """
    Calculate the footprint climatology for an eddy covariance flux station.

    This function retrieves flux and meteorological data from a database, processes the data,
    and calculates footprint climatology for each valid day. The results are stored as GeoTIFF
    raster files representing daily footprints.

    Parameters:
    ----------
    station : str
        Station identifier used to retrieve configuration and observational data.
    startdate : str, optional (default: '2022-01-01')
        The start date for querying flux and meteorological data.
    config_path : str, optional
        Path to the station configuration file containing metadata.
    secrets_path : str, optional
        Path to the secrets configuration file containing database credentials.
    epsg : int, optional (default: 5070)
        EPSG code for coordinate transformation (default is NAD83/Conus Albers).
    h_c : float, optional (default: 0.2)
        Height of the canopy in meters.
    zm_s : float, optional (default: 2.0)
        Measurement height above ground in meters.
    dx : float, optional (default: 3.0)
        Grid resolution for the footprint model in meters.
    h_s : float, optional (default: 2000.0)
        Assumed height of the atmospheric boundary layer in meters.
    origin_d : float, optional (default: 200.0)
        Distance from the origin defining the model domain in meters.

    Process:
    --------
    1. Loads station metadata (latitude, longitude, elevation) from a configuration file.
    2. Retrieves flux data (`amfluxeddy`) from the database and preprocesses it:
       - Converts timestamps to datetime format.
       - Resamples data to hourly intervals.
       - Filters out invalid or missing values.
    3. Converts station coordinates from latitude/longitude to UTM using the specified EPSG code.
    4. Computes displacement height (`d`) and adjusts the measurement height (`zm`).
    5. Loops through each valid day and processes hourly footprint fields:
       - Calls `ffp.ffp_climatology` to compute footprint climatology.
       - Transforms footprint grid coordinates.
       - Creates an affine transformation for georeferencing.
    6. Writes the output to a multi-band GeoTIFF file, where each band represents an hourly footprint.
    7. Closes the dataset safely after writing all footprint data.

    Output:
    -------
    - GeoTIFF Raster File: `{YYYY-MM-DD}_weighted.tif`
      - Stores footprint climatology for each valid day.
      - Georeferenced to the station’s UTM coordinates.

    Dependencies:
    -------------
    - configparser
    - requests
    - pandas
    - numpy
    - pathlib
    - pyproj
    - rasterio
    - micromet
    - ffp (Footprint modeling library)
    - multiprocessing

    Example Usage:
    --------------
    >>> calc_ffp('US-UTW')

    Parallel Execution:
    -------------------
    The function can be executed in parallel for multiple stations using multiprocessing:

    >>> pool = mp.Pool(processes=8)
    >>> pool.map(calc_ffp, ['US-UTW', 'US-XYZ', 'US-ABC'])

    Notes:
    ------
    - The function skips days with fewer than 5 valid hourly records.
    - Existing daily raster files are not overwritten.
    - Only data from 6 AM to 8 PM is used for footprint calculations.
    - Errors during hourly footprint computations are handled gracefully, skipping affected hours.
    """
    if config_path is None:
        config_path = f'../../station_config/{station}.ini'

    metadata = _load_configs(station, config_path, secrets_path)
    df = _fetch_and_preprocess_data(metadata["url"], station, startdate)
    if df.empty:
        logging.error(f"No valid data found for station {station}. Skipping.")
        return

    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}")
    station_y, station_x = transformer.transform(metadata["latitude"], metadata["longitude"])

    d = 10 ** (0.979 * np.log10(h_c) - 0.154)
    zm = zm_s - d
    z0 = h_c * 0.123

    out_dir = pathlib.Path('.')

    for date in df.index.date:
        temp_df = df[df.index.date == date].between_time("06:00", "20:00")
        if len(temp_df) < 5:
            logging.warning(f"Less than 5 hours of data on {date}, skipping.")
            continue

        final_outf = out_dir / f'{date}_weighted.tif'
        if final_outf.is_file():
            logging.info(f"Final weighted footprint already exists: {final_outf}. Skipping.")
            continue

        footprints = _compute_hourly_footprint(temp_df, station_x, station_y, zm, h_s, z0, dx, origin_d)
        if footprints:
            _write_footprint_to_raster(footprints, final_outf, epsg=epsg)



def mask_fp_cutoff(f_array, cutoff=0.9):
    """
    Masks all values outside of the cutoff value

    Args:
        f_array (float) : 2D numpy array of point footprint contribution values (no units)
        cutoff (float) : Cutoff value for the cumulative sum of footprint values

    Returns:
        f_array (float) : 2D numpy array of footprint values, with nan == 0
    """
    val_array = f_array.flatten()
    sort_df = pd.DataFrame({"f": val_array}).sort_values(by="f").iloc[::-1]
    sort_df["cumsum_f"] = sort_df["f"].cumsum()

    sort_group = sort_df.groupby("f", as_index=True).mean()
    diff = abs(sort_group["cumsum_f"] - cutoff)
    sum_cutoff = diff.idxmin()
    f_array = np.where(f_array >= sum_cutoff, f_array, np.nan)
    f_array[~np.isfinite(f_array)] = 0.00000000e000

    logger.debug(f"mask_fp_cutoff: applied cutoff={cutoff}, sum_cutoff={sum_cutoff}")
    return f_array


def find_transform(xs, ys):
    """
    Returns the affine transform for 2d arrays xs and ys

    Args:
        xs (float) : 2D numpy array of x-coordinates
        ys (float) : 2D numpy array of y-coordinates

    Returns:
        aff_transform : affine.Affine object
    """

    shape = xs.shape

    # Choose points to calculate affine transform
    y_points = [0, 0, shape[0] - 1]
    x_points = [0, shape[0] - 1, shape[1] - 1]
    in_xy = np.float32([[i, j] for i, j in zip(x_points, y_points)])
    out_xy = np.float32([[xs[i, j], ys[i, j]] for i, j in zip(y_points, x_points)])

    # Calculate affine transform
    aff_transform = Affine(*cv2.getAffineTransform(in_xy, out_xy).flatten())
    logger.debug("Affine transform calculated.")
    return aff_transform


def ffp_climatology(
    zm=None,
    z0=None,
    umean=None,
    h=None,
    ol=None,
    sigmav=None,
    ustar=None,
    wind_dir=None,
    domain=None,
    dx=None,
    dy=None,
    nx=None,
    ny=None,
    rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    rslayer=0,
    smooth_data=1,
    crop=False,
    pulse=None,
    verbosity=2,
    fig=False,
    **kwargs,
):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: n.kljun@swansea.ac.uk

    This function calculates footprints within a fixed physical domain for a series of
    time steps, rotates footprints into the corresponding wind direction and aggregates
    all footprints to a footprint climatology. The percentage of source area is
    calculated for the footprint climatology.
    For determining the optimal extent of the domain (large enough to include footprints)
    use calc_footprint_FFP.py.

    FFP Input
        All vectors need to be of equal length (one value for each time step)
        zm       = Measurement height above displacement height (i.e. z-d) [m]
                   usually a scalar, but can also be a vector
        z0       = Roughness length [m] - enter [None] if not known
                   usually a scalar, but can also be a vector
        umean    = Vector of mean wind speed at zm [ms-1] - enter [None] if not known
                   Either z0 or umean is required. If both are given,
                   z0 is selected to calculate the footprint
        h        = Vector of boundary layer height [m]
        ol       = Vector of Obukhov length [m]
        sigmav   = Vector of standard deviation of lateral velocity fluctuations [ms-1]
        ustar    = Vector of friction velocity [ms-1]
        wind_dir = Vector of wind direction in degrees (of 360) for rotation of the footprint

        Optional input:
        domain       = Domain size as an array of [xmin xmax ymin ymax] [m].
                       Footprint will be calculated for a measurement at [0 0 zm] m
                       Default is smallest area including the r% footprint or [-1000 1000 -1000 1000]m,
                       whichever smallest (80% footprint if r not given).
        dx, dy       = Cell size of domain [m]
                       Small dx, dy results in higher spatial resolution and higher computing time
                       Default is dx = dy = 2 m. If only dx is given, dx=dy.
        nx, ny       = Two integer scalars defining the number of grid elements in x and y
                       Large nx/ny result in higher spatial resolution and higher computing time
                       Default is nx = ny = 1000. If only nx is given, nx=ny.
                       If both dx/dy and nx/ny are given, dx/dy is given priority if the domain is also specified.
        rs           = Percentage of source area for which to provide contours, must be between 10% and 90%.
                       Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
                       Expressed either in percentages ("80") or as fractions of 1 ("0.8").
                       Default is [10:10:80]. Set to "None" for no output of percentages
        rslayer      = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
                       Note that this only gives a rough estimate of the footprint as the model is not
                       valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
                       z0 is needed for estimation of the RS.
        smooth_data  = Apply convolution filter to smooth footprint climatology if smooth_data=1 (default)
        crop         = Crop output area to size of the 80% footprint or the largest r given if crop=1
        pulse        = Display progress of footprint calculations every pulse-th footprint (e.g., "100")
        verbosity    = Level of verbosity at run time: 0 = completely silent, 1 = notify only of fatal errors,
                       2 = all notifications
        fig          = Plot an example figure of the resulting footprint (on the screen): set fig = 1.
                       Default is 0 (i.e. no figure).

    FFP output
        FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
        x_2d	    = x-grid of 2-dimensional footprint [m]
        y_2d	    = y-grid of 2-dimensional footprint [m]
        fclim_2d = Normalised footprint function values of footprint climatology [m-2]
        rs       = Percentage of footprint as in input, if provided
        fr       = Footprint value at r, if r is provided
        xr       = x-array for contour line of r, if r is provided
        yr       = y-array for contour line of r, if r is provided
        n        = Number of footprints calculated and included in footprint climatology
        flag_err = 0 if no error, 1 in case of error, 2 if not all contour plots (rs%) within specified domain,
                   3 if single data points had to be removed (outside validity)

    Created: 19 May 2016 natascha kljun
    Converted from matlab to python, together with Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.4
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015,2016,2017,2018,2019,2020 Natascha Kljun
    """

    # ===========================================================================
    # Get kwargs
    show_heatmap = kwargs.get("show_heatmap", True)

    # ===========================================================================
    # Input check
    flag_err = 0

    # Check existence of required input pars
    if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1, verbosity)

    # List of variables to be converted to lists
    variables = [zm, h, ol, sigmav, ustar, wind_dir, z0, umean]

    # Convert each variable to a list if it is not already a list
    variables = [[var] if not isinstance(var, list) else var for var in variables]

    # Unpack back into individual variables
    zm, h, ol, sigmav, ustar, wind_dir, z0, umean = variables

    # Check that all lists have same length, if not raise an error and exit
    ts_len = len(ustar)

    logger.debug(f"input len is {ts_len}")

    if any(len(lst) != ts_len for lst in [sigmav, wind_dir, h, ol]):
        # at least one list has a different length, exit with error message
        raise_ffp_exception(11, verbosity)

    # Special treatment for zm, which is allowed to have length 1 for any
    # length >= 1 of all other parameters
    if all(val is None for val in zm):
        raise_ffp_exception(12, verbosity)
    if len(zm) == 1:
        raise_ffp_exception(17, verbosity)
        zm = [zm[0] for i in range(ts_len)]

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    # If at least one value of z0 is passed, use z0 (by setting umean to None)
    if not all(val is None for val in z0):
        raise_ffp_exception(13, verbosity)
        umean = [None for i in range(ts_len)]
        # If only one value of z0 was passed, use that value for all footprints
        if len(z0) == 1:
            z0 = [z0[0] for i in range(ts_len)]
    elif len(umean) == ts_len and not all(val is None for val in umean):
        raise_ffp_exception(14, verbosity)
        z0 = [None for i in range(ts_len)]
    else:
        raise_ffp_exception(15, verbosity)

    # Rename lists as now the function expects time series of inputs
    ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans = (
        ustar,
        sigmav,
        h,
        ol,
        wind_dir,
        zm,
        z0,
        umean,
    )

    logger.debug(
        f"variables ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans input: {ustars}, {sigmavs}, {hs}, {ols}, {wind_dirs}, {zms}, {z0s}, {umeans}"
    )

    # ===========================================================================
    # Handle rs
    if rs is not None:
        # Check that rs is a list, otherwise make it a list
        if isinstance(rs, numbers.Number):
            if 0.9 < rs <= 1 or 90 < rs <= 100:
                rs = 0.9
            rs = [rs]
        if not isinstance(rs, list):
            raise_ffp_exception(18, verbosity)

        # If rs is passed as percentages, normalize to fractions of one
        if np.max(rs) >= 1:
            rs = [x / 100.0 for x in rs]

        # Eliminate any values beyond 0.9 (90%) and inform user
        if np.max(rs) > 0.9:
            raise_ffp_exception(19, verbosity)
            rs = [item for item in rs if item <= 0.9]

        # Sort levels in ascending order
        rs = list(np.sort(rs))

    # ===========================================================================
    # Define computational domain
    # Check passed values and make some smart assumptions
    if isinstance(dx, numbers.Number) and dy is None:
        dy = dx
    if isinstance(dy, numbers.Number) and dx is None:
        dx = dy
    if not all(isinstance(item, numbers.Number) for item in [dx, dy]):
        dx = dy = None
    if isinstance(nx, int) and ny is None:
        ny = nx
    if isinstance(ny, int) and nx is None:
        nx = ny
    if not all(isinstance(item, int) for item in [nx, ny]):
        nx = ny = None
    if not isinstance(domain, list) or len(domain) != 4:
        domain = None

    if all(item is None for item in [dx, nx, domain]):
        # If nothing is passed, default domain is a square of 2 Km size centered
        # at the tower with pizel size of 2 meters (hence a 1000x1000 grid)
        domain = [-1000.0, 1000.0, -1000.0, 1000.0]
        dx = dy = 2.0
        nx = ny = 1000
    elif domain is not None:
        # If domain is passed, it takes the precendence over anything else
        if dx is not None:
            # If dx/dy is passed, takes precendence over nx/ny
            nx = int((domain[1] - domain[0]) / dx)
            ny = int((domain[3] - domain[2]) / dy)
        else:
            # If dx/dy is not passed, use nx/ny (set to 1000 if not passed)
            if nx is None:
                nx = ny = 1000
            # If dx/dy is not passed, use nx/ny
            dx = (domain[1] - domain[0]) / float(nx)
            dy = (domain[3] - domain[2]) / float(ny)
    elif dx is not None and nx is not None:
        # If domain is not passed but dx/dy and nx/ny are, define domain
        domain = [-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2]
    elif dx is not None:
        # If domain is not passed but dx/dy is, define domain and nx/ny
        domain = [-1000, 1000, -1000, 1000]
        nx = int((domain[1] - domain[0]) / dx)
        ny = int((domain[3] - domain[2]) / dy)
    elif nx is not None:
        # If domain and dx/dy are not passed but nx/ny is, define domain and dx/dy
        domain = [-1000, 1000, -1000, 1000]
        dx = (domain[1] - domain[0]) / float(nx)
        dy = (domain[3] - domain[2]) / float(nx)

    # Put domain into more convenient vars
    xmin, xmax, ymin, ymax = domain
    logger.info(f"Domain: {domain}")
    # Define rslayer if not passed
    if rslayer is None:
        rslayer = 0

    # Define smooth_data if not passed
    if smooth_data is None:
        smooth_data = 1

    # Define crop if not passed
    if crop is None:
        crop = 0

    # Define pulse if not passed
    if pulse is None:
        if ts_len <= 20:
            pulse = 1
        else:
            pulse = int(ts_len / 20)

    # Define fig if not passed
    if fig is None:
        fig = 0

    logger.debug(
        f"parameters rslayer, smooth_data, crop, pulse, fig: {rslayer}, {smooth_data}, {crop}, {pulse}, {fig}"
    )

    # ===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    oln = 5000  # limit to L for neutral scaling
    k = 0.4  # von Karman

    # ===========================================================================
    # Define physical domain in cartesian and polar coordinates
    # Cartesian coordinates
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    x_2d, y_2d = np.meshgrid(x, y)
    logger.debug(f"x_2d: {x_2d}, y_2d: {y_2d}")
    # Polar coordinates
    # Set theta such that North is pointing upwards and angles increase clockwise
    rho = np.sqrt(x_2d**2 + y_2d**2)
    theta = np.arctan2(x_2d, y_2d)
    logger.debug(f"rho: {rho}, theta: {theta}")
    # initialize raster for footprint climatology
    fclim_2d = np.zeros(x_2d.shape)

    # ===========================================================================
    # Loop on time series

    # Initialize logic array valids to those 'timestamps' for which all inputs are
    # at least present (but not necessarily phisically plausible)
    valids = [
        True if not any([val is None for val in vals]) else False
        for vals in zip(ustars, sigmavs, hs, ols, wind_dirs, zms)
    ]

    logger.debug(f"List of valids {valids}")

    if verbosity > 1:
        logger.info("Beginning footprint calculations...")

    for ix, (ustar, sigmav, h, ol, wind_dir, zm, z0, umean) in enumerate(
        zip(ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans)
    ):
        # Counter
        if verbosity > 1 and ix % pulse == 0:
            print("Calculating footprint ", ix + 1, " of ", ts_len)
            logger.info(f"Calculating footprint {ix + 1} of {ts_len}")

        valids[ix] = check_ffp_inputs(
            ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbosity
        )

        logger.debug(f"valids of {ix} are {valids[ix]}")

        # If inputs are not valid, skip current footprint
        if not valids[ix]:
            raise_ffp_exception(16, verbosity)
        else:
            # ===========================================================================
            # Rotate coordinates into wind direction
            if wind_dir is not None:
                rotated_theta = theta - wind_dir * np.pi / 180.0

                logger.debug(f"rotated_theta: {rotated_theta}")
            # ===========================================================================
            # Create real scale crosswind integrated footprint and dummy for
            # rotated scaled footprint
            fstar_ci_dummy = np.zeros(x_2d.shape)
            f_ci_dummy = np.zeros(x_2d.shape)
            xstar_ci_dummy = np.zeros(x_2d.shape)
            px = np.ones(x_2d.shape)

            if z0 is not None:
                # Use z0
                if ol <= 0 or ol >= oln:
                    xx = (1 - 19.0 * zm / ol) ** 0.25
                    psi_f = (
                        np.log((1 + xx**2) / 2.0)
                        + 2.0 * np.log((1 + xx) / 2.0)
                        - 2.0 * np.arctan(xx)
                        + np.pi / 2
                    )
                    logger.debug(f"psi_f = {psi_f}, xx = {xx}")
                elif ol > 0 and ol < oln:
                    psi_f = -5.3 * zm / ol
                    # print(psi_f, zm, ol)
                    logger.debug(f"psi_f = {psi_f}, zm = {zm}, ol = {ol}")

                if (np.log(zm / z0) - psi_f) > 0:
                    logger.debug("Calculating xstar_ci_dummy...")
                    xstar_ci_dummy = (
                        rho
                        * np.cos(rotated_theta)
                        / zm
                        * (1.0 - (zm / h))
                        / (np.log(zm / z0) - psi_f)
                    )
                    px = np.where(xstar_ci_dummy > d)
                    fstar_ci_dummy[px] = (
                        a
                        * (xstar_ci_dummy[px] - d) ** b
                        * np.exp(-c / (xstar_ci_dummy[px] - d))
                    )
                    f_ci_dummy[px] = (
                        fstar_ci_dummy[px]
                        / zm
                        * (1.0 - (zm / h))
                        / (np.log(zm / z0) - psi_f)
                    )

                else:
                    flag_err = 3
                    valids[ix] = 0
                    logger.debug("flag err 3")
            else:
                # Use umean if z0 not available
                xstar_ci_dummy = (
                    rho
                    * np.cos(rotated_theta)
                    / zm
                    * (1.0 - (zm / h))
                    / (umean / ustar * k)
                )
                px = np.where(xstar_ci_dummy > d)
                fstar_ci_dummy[px] = (
                    a
                    * (xstar_ci_dummy[px] - d) ** b
                    * np.exp(-c / (xstar_ci_dummy[px] - d))
                )
                f_ci_dummy[px] = (
                    fstar_ci_dummy[px] / zm * (1.0 - (zm / h)) / (umean / ustar * k)
                )

            # ===========================================================================
            # Calculate dummy for scaled sig_y* and real scale sig_y
            sigystar_dummy = np.zeros(x_2d.shape)
            sigystar_dummy[px] = ac * np.sqrt(
                bc
                * np.abs(xstar_ci_dummy[px]) ** 2
                / (1 + cc * np.abs(xstar_ci_dummy[px]))
            )

            if abs(ol) > oln:
                ol = -1e6
            if ol <= 0:  # convective
                scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.80
            elif ol > 0:  # stable
                scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.55
            if scale_const > 1:
                scale_const = 1.0

            sigy_dummy = np.zeros(x_2d.shape)
            sigy_dummy[px] = sigystar_dummy[px] / scale_const * zm * sigmav / ustar
            sigy_dummy[sigy_dummy < 0] = np.nan

            # ===========================================================================
            # Calculate real scale f(x,y)
            f_2d = np.zeros(x_2d.shape)
            f_2d[px] = (
                f_ci_dummy[px]
                / (np.sqrt(2 * np.pi) * sigy_dummy[px])
                * np.exp(
                    -((rho[px] * np.sin(rotated_theta[px])) ** 2)
                    / (2.0 * sigy_dummy[px] ** 2)
                )
            )

            # ===========================================================================
            # Add to footprint climatology raster
            fclim_2d = fclim_2d + f_2d
            logger.debug(f"fclim_2d: {fclim_2d}, f_2d: {f_2d}")
    # ===========================================================================
    # Continue if at least one valid footprint was calculated
    n = sum(valids)
    logger.debug(f"n: {n}")
    vs = None
    clevs = None
    if n == 0:
        logger.warning("No valid footprints were calculated.")
        print("No footprint calculated")
        flag_err = 1
    else:

        # ===========================================================================
        # Normalize and smooth footprint climatology
        fclim_2d = fclim_2d / n

        if smooth_data is not None:
            skernel = np.matrix("0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05")
            fclim_2d = sg.convolve2d(fclim_2d, skernel, mode="same")
            fclim_2d = sg.convolve2d(fclim_2d, skernel, mode="same")

        # ===========================================================================
        # Derive footprint ellipsoid incorporating R% of the flux, if requested,
        # starting at peak value.
        if rs is not None:
            clevs = get_contour_levels(fclim_2d, dx, dy, rs)
            frs = [item[2] for item in clevs]
            xrs = []
            yrs = []
            for ix, fr in enumerate(frs):
                xr, yr = get_contour_vertices(x_2d, y_2d, fclim_2d, fr)
                if xr is None:
                    frs[ix] = None
                    flag_err = 2
                xrs.append(xr)
                yrs.append(yr)
        else:
            if crop:
                rs_dummy = 0.8  # crop to 80%
                clevs = get_contour_levels(fclim_2d, dx, dy, rs_dummy)
                xrs = []
                yrs = []
                xrs, yrs = get_contour_vertices(x_2d, y_2d, fclim_2d, clevs[0][2])

        # ===========================================================================
        # Crop domain and footprint to the largest rs value
        if crop:
            xrs_crop = [x for x in xrs if x is not None]
            yrs_crop = [x for x in yrs if x is not None]
            if rs is not None:
                dminx = np.floor(min(xrs_crop[-1]))
                dmaxx = np.ceil(max(xrs_crop[-1]))
                dminy = np.floor(min(yrs_crop[-1]))
                dmaxy = np.ceil(max(yrs_crop[-1]))
            else:
                dminx = np.floor(min(xrs_crop))
                dmaxx = np.ceil(max(xrs_crop))
                dminy = np.floor(min(yrs_crop))
                dmaxy = np.ceil(max(yrs_crop))

            if dminy >= ymin and dmaxy <= ymax:
                jrange = np.where((y_2d[:, 0] >= dminy) & (y_2d[:, 0] <= dmaxy))[0]
                jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
                jrange = jrange[np.where((jrange >= 0) & (jrange <= y_2d.shape[0]))[0]]
            else:
                jrange = np.linspace(0, 1, y_2d.shape[0] - 1)

            if dminx >= xmin and dmaxx <= xmax:
                irange = np.where((x_2d[0, :] >= dminx) & (x_2d[0, :] <= dmaxx))[0]
                irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
                irange = irange[np.where((irange >= 0) & (irange <= x_2d.shape[1]))[0]]
            else:
                irange = np.linspace(0, 1, x_2d.shape[1] - 1)

            jrange = [[it] for it in jrange]
            x_2d = x_2d[jrange, irange]
            y_2d = y_2d[jrange, irange]
            fclim_2d = fclim_2d[jrange, irange]

        # ===========================================================================
        # Plot footprint
        if fig:
            fig_out, ax = plot_footprint(
                x_2d=x_2d, y_2d=y_2d, fs=fclim_2d, show_heatmap=show_heatmap, clevs=frs
            )
        else:
            fig_out = None

    # ===========================================================================
    # Fill output structure
    if rs is not None:
        return {
            "x_2d": x_2d,
            "y_2d": y_2d,
            "fclim_2d": fclim_2d,
            "rs": rs,
            "fr": frs,
            "xr": xrs,
            "yr": yrs,
            "n": n,
            "flag_err": flag_err,
            "fig": fig_out,
        }
    else:
        return {
            "x_2d": x_2d,
            "y_2d": y_2d,
            "fclim_2d": fclim_2d,
            "n": n,
            "flag_err": flag_err,
        }  # 'fig': fig_out


def check_ffp_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbosity):
    """
    Validates input parameters for physical plausibility and consistency for a footprint model.

    This function checks the validity of the input parameters provided for a footprint
    model. It ensures that the parameters adhere to the required physical constraints
    and consistency rules. The function raises exceptions when specific conditions
    are violated and can also operate with a specified verbosity level for raising
    exceptions.

    Args:
        ustar (float or array-like): Friction velocity. The value must be greater than 0.1
            for all elements.
        sigmav (float or array-like): Standard deviation of vertical wind speed. The value
            must be greater than 0 for all elements.
        h (float): Boundary layer height. The value must be greater than 10.
        ol (float or array-like): Obukhov length. The computed ratio of `zm / ol` must
            not be less than -15.5.
        wind_dir (float or array-like): Wind direction in degrees. Must be in the
            range [0, 360] for all elements.
        zm (float): Measurement height above ground. Must be positive and less than the
            boundary layer height (`h`).
        z0 (float, optional): Surface roughness length. If specified, must be greater
            than 0.
        umean (float, optional): Mean wind speed. Required if `z0` is specified.
        rslayer (int): Stability regime flag. Can modify the criteria for how `zm` and `z0`
            are validated.
        verbosity (int): Verbosity level for exception handling. Higher values may allow
            more detailed error reporting.

    Returns:
        bool: True if all inputs pass the validity checks, otherwise raises an exception.

    Raises:
        Exception: Raised with specific error codes for violations of physical validity:
            - Code 2: Measurement height (`zm`) is non-positive.
            - Code 3: Surface roughness length (`z0`) is non-positive without `umean`.
            - Code 4: Boundary layer height (`h`) is too small (≤ 10 m).
            - Code 5: Measurement height (`zm`) exceeds boundary layer height (`h`).
            - Code 6 or 20: Measurement height (`zm`) is inconsistent with surface
              roughness length (`z0`) given the stability layer regime.
            - Code 7: Ratio of measurement height to Obukhov length (`zm / ol`) is
              less than -15.5.
            - Code 8: Standard deviation of vertical wind speed (`sigmav`) is
              non-positive.
            - Code 9: Friction velocity (`ustar`) is too small (≤ 0.1).
            - Code 10: Wind direction values are out of the valid range ([0, 360]).
    """
    # Check passed values for physical plausibility and consistency
    if zm <= 0.0:
        raise_ffp_exception(2, verbosity)
        logger.debug(f"zm <= 0.0   zm={zm}")
        return False
    if z0 is not None and umean is None and z0 <= 0.0:
        raise_ffp_exception(3, verbosity)
        logger.debug("z0 is not None and umean is None and z0 <= 0.0")
        return False
    if h <= 10.0:
        raise_ffp_exception(4, verbosity)
        logger.debug(f"h <= 10.0  h={h}")
        return False
    if zm > h:
        raise_ffp_exception(5, verbosity)
        logger.debug(f"zm > h  zm={zm}, h={h}")
        return False
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        logger.debug(f"zm <= 12.5 * z0   zm={zm}, z0={z0}")
        if rslayer == 1:
            raise_ffp_exception(6, verbosity)
            logger.debug("rslayer == 1")
        else:
            raise_ffp_exception(20, verbosity)
            logger.debug("rslayer != 1")
            return False
    if (float(zm) / ol).any() <= -15.5:
        raise_ffp_exception(7, verbosity)
        logger.debug(f"float(zm) / ol).any() <= -15.5  zm={zm}, ol={ol}")
        return False
    if sigmav.any() <= 0:
        raise_ffp_exception(8, verbosity)
        logger.debug(f"sigmav.any() <= 0  sigmav={sigmav}")
        return False
    if ustar.any() <= 0.1:
        raise_ffp_exception(9, verbosity)
        logger.debug(f"ustar.any() <= 0.1 {ustar}")
        return False
    if wind_dir.any() > 360:
        raise_ffp_exception(10, verbosity)
        logger.debug(f"wind_dir.any() > 360   wind_dir={wind_dir}")
        return False
    if wind_dir.any() < 0:
        logger.debug(f"wind_dir.any() < 0  wind_dir={wind_dir}")
        raise_ffp_exception(10, verbosity)
        return False
    return True


def get_contour_levels(f, dx, dy, rs=None):
    """
    Computes contour levels for a given array of values based on specified resolution and area fractions.

    This function is designed to determine levels for contour plotting by
    associating them with cumulative area fractions of sorted field values (`f`).

    Args:
        f (np.ndarray): The 2D array of field values for which contour levels are
            to be calculated. Array can contain NaN or infinity values, which will
            be properly handled during computation.
        dx (float): The resolution or cell size in the x-direction. Represents the
            grid spacing along the horizontal axis.
        dy (float): The resolution or cell size in the y-direction. Represents the
            grid spacing along the vertical axis.
        rs (Optional[Union[float, int, List[Union[float, int]]]]): The fractional
            areas (values between 0 and 1) for which contour levels need to be
            computed. Defaults to nine equally spaced fractions between 0.1 and
            0.9 if not provided. If `rs` is a single number, it will be converted
            into a list.

    Returns:
        List[Tuple[float, float, float]]: A list of tuples with each tuple
        containing:
            - The fractional area (rounded to three decimal places).
            - The cumulative area value at the corresponding contour level.
            - The corresponding contour level value from the input field `f`
              associated with that fractional area.

    Raises:
        ValueError: If the input `rs` is not of acceptable types (`int`, `float`,
            or `list`) or does not fall into the range between 0 and 1.
    """

    # Check input and resolve to default levels in needed
    if not isinstance(rs, (int, float, list)):
        rs = list(np.linspace(0.10, 0.90, 9))
    if isinstance(rs, (int, float)):
        rs = [rs]

    # Levels
    pclevs = np.empty(len(rs))
    pclevs[:] = np.nan
    ars = np.empty(len(rs))
    ars[:] = np.nan
    logger.debug(pclevs)

    sf = np.sort(f, axis=None)[::-1]
    msf = ma.masked_array(
        sf, mask=(np.isnan(sf) | np.isinf(sf))
    )  # Masked array for handling potential nan
    csf = msf.cumsum().filled(np.nan) * dx * dy
    for ix, r in enumerate(rs):
        dcsf = np.abs(csf - r)
        pclevs[ix] = sf[np.nanargmin(dcsf)]
        ars[ix] = csf[np.nanargmin(dcsf)]

    return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]


def get_contour_vertices(x, y, f, lev):
    # import matplotlib._contour as cntr
    import matplotlib.pyplot as plt

    cs = plt.contour(x, y, f, [lev])
    plt.close()
    segs = cs.allsegs[0][0]
    logger.debug(segs)
    xr = [vert[0] for vert in segs]
    yr = [vert[1] for vert in segs]
    # Set contour to None if it's found to reach the physical domain
    if (
        x.min() >= min(segs[:, 0])
        or max(segs[:, 0]) >= x.max()
        or y.min() >= min(segs[:, 1])
        or max(segs[:, 1]) >= y.max()
    ):
        return [None, None]

    return [xr, yr]  # x,y coords of contour points.


def plot_footprint(
    x_2d,
    y_2d,
    fs,
    clevs=None,
    show_heatmap=True,
    normalize=None,
    colormap=None,
    line_width=0.5,
    iso_labels=None,
):
    """
    Plots footprint data and optionally overlays contours, isopleths, and heatmaps for one
    or more footprints over a 2D grid.

    This function visualizes footprint data by creating either a heatmap, contour plot, or
    both, depending on the provided parameters. It supports multiple footprints, each
    represented by a unique contour color when overlaid together. Customization options
    are available for contour levels, colormap, line width, isopleth labels, and normalization.

    Args:
        x_2d (numpy.ndarray): 2D array representing the x-coordinates of the grid.
        y_2d (numpy.ndarray): 2D array representing the y-coordinates of the grid.
        fs (numpy.ndarray or list of numpy.ndarray): Footprint data as a 2D array for single
            footprint or list of 2D arrays for multiple footprints.
        clevs (list[float] or None): Contour levels for the plot. If None, no contours are
            drawn. Defaults to None.
        show_heatmap (bool): If True, displays a heatmap for the footprint. Defaults to True.
        normalize (str or None): Normalization method for the heatmap. Specify "log" for
            logarithmic normalization, or None for no normalization. Defaults to None.
        colormap (matplotlib.colors.Colormap or None): Colormap to use for plotting the
            heatmap or contours. Defaults to None, which uses `cm.jet`.
        line_width (float): Line width for contour plotting. Defaults to 0.5.
        iso_labels (list[tuple[float]] or None): Labels for the isopleths as percentages.
            If None, no isopleth labels are added. Defaults to None.

    Returns:
        tuple: Contains the following:
            - fig (matplotlib.figure.Figure): The figure object containing the plot.
            - ax (matplotlib.axes.Axes): The axes object of the plot.
    """

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    if colormap is None:
        colormap = cm.jet
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 8))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs is not None:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = clevs[::-1]

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev is not None]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c] * len(levs)
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors="w", linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors=cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels is not None:
                pers = [str(int(clev[0] * 100)) + "%" for clev in clevs]
                fmt = {}
                for l, s in zip(cp.levels, pers):
                    fmt[l] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_heatmap:
        if normalize == "log":
            norm = LogNorm()
        else:
            norm = None

        xmin = np.nanmin(x_2d)
        xmax = np.nanmax(x_2d)
        ymin = np.nanmin(y_2d)
        ymax = np.nanmax(y_2d)
        for f in fs:
            im = ax.imshow(
                f[:, :],
                cmap=colormap,
                extent=(xmin, xmax, ymin, ymax),
                norm=norm,
                origin="lower",
                aspect=1,
            )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        # Colorbar
        cbar = fig.colorbar(im, shrink=1.0, format="%.3e")
        # cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax


exTypes = {
    "message": "Message",
    "alert": "Alert",
    "error": "Error",
    "fatal": "Fatal error",
}

exceptions = [
    {
        "code": 1,
        "type": exTypes["fatal"],
        "msg": "At least one required parameter is missing. Please enter all "
        "required inputs. Check documentation for details.",
    },
    {
        "code": 2,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be larger than zero.",
    },
    {
        "code": 3,
        "type": exTypes["error"],
        "msg": "z0 (roughness length) must be larger than zero.",
    },
    {
        "code": 4,
        "type": exTypes["error"],
        "msg": "h (BPL height) must be larger than 10 m.",
    },
    {
        "code": 5,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be smaller than h (PBL height).",
    },
    {
        "code": 6,
        "type": exTypes["alert"],
        "msg": "zm (measurement height) should be above roughness sub-layer (12.5*z0).",
    },
    {
        "code": 7,
        "type": exTypes["error"],
        "msg": "zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.",
    },
    {
        "code": 8,
        "type": exTypes["error"],
        "msg": "sigmav (standard deviation of crosswind) must be larger than zero.",
    },
    {
        "code": 9,
        "type": exTypes["error"],
        "msg": "ustar (friction velocity) must be >=0.1.",
    },
    {
        "code": 10,
        "type": exTypes["error"],
        "msg": "wind_dir (wind direction) must be >=0 and <=360.",
    },
    {
        "code": 11,
        "type": exTypes["fatal"],
        "msg": "Passed data arrays (ustar, zm, h, ol) don't all have the same length.",
    },
    {
        "code": 12,
        "type": exTypes["fatal"],
        "msg": "No valid zm (measurement height above displacement height) passed.",
    },
    {
        "code": 13,
        "type": exTypes["alert"],
        "msg": "Using z0, ignoring umean if passed.",
    },
    {"code": 14, "type": exTypes["alert"], "msg": "No valid z0 passed, using umean."},
    {"code": 15, "type": exTypes["fatal"], "msg": "No valid z0 or umean array passed."},
    {
        "code": 16,
        "type": exTypes["error"],
        "msg": "At least one required input is invalid. Skipping current footprint.",
    },
    {
        "code": 17,
        "type": exTypes["alert"],
        "msg": "Only one value of zm passed. Using it for all footprints.",
    },
    {
        "code": 18,
        "type": exTypes["fatal"],
        "msg": "if provided, rs must be in the form of a number or a list of numbers.",
    },
    {
        "code": 19,
        "type": exTypes["alert"],
        "msg": "rs value(s) larger than 90% were found and eliminated.",
    },
    {
        "code": 20,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be above roughness sub-layer (12.5*z0).",
    },
]


def raise_ffp_exception(code, verbosity):
    """
    Raises exceptions based on provided error code and verbosity level, with appropriate logging
    and messaging defined by the exception type.

    The function utilizes an external `exceptions` list to locate the matching exception type
    and message using the provided error code. Depending on the verbosity level and exception
    type, it either logs or prints warnings and errors or raises a critical exception, signaling
    an immediate program halt. The exception messaging can be customized based on execution
    needs and conditions.

    Args:
        code: An integer representing the error code used to fetch details about a specific
            exception type and its corresponding message.
        verbosity: An integer controlling the level of detail in exception logging and messaging.
            Higher verbosity levels include more detailed output, with level 0 suppressing most
            message outputs.

    Raises:
        Exception: Raised when the exception type corresponds to a fatal error. The execution of
            the program is forcibly aborted in such cases.
    """

    ex = [it for it in exceptions if it["code"] == code][0]
    string = ex["type"] + "(" + str(ex["code"]).zfill(4) + "):\n " + ex["msg"]

    if verbosity > 0:
        logger.warning("")

    if ex["type"] == exTypes["fatal"]:
        if verbosity > 0:
            string = string + "\n FFP_fixed_domain execution aborted."
            logger.error(string)
        else:
            string = ""
        raise Exception(string)
    elif ex["type"] == exTypes["alert"]:
        string = string + "\n Execution continues."
        if verbosity > 1:
            print(string)
            logger.warning(string)
    elif ex["type"] == exTypes["error"]:
        string = string + "\n Execution continues."
        if verbosity > 1:
            print(string)
            logger.error(string)
    else:
        if verbosity > 1:
            print(string)
            logger.warning(string)


def download_nldas(date,
                   hour,
                   ed_user,
                   ed_pass,):

    #NLDAS version 2, primary forcing set (a), DOY must be 3 digit zero padded, HH 2-digit between 00-23, MM and DD also 2 digit
    YYYY = date.year
    DOY = date.timetuple().tm_yday
    MM = date.month
    DD = date.day
    HH = hour

    nldas_out_dir = Path('NLDAS_data')
    if not nldas_out_dir.is_dir():
        nldas_out_dir.mkdir(parents=True, exist_ok=True)

    nldas_outf_path = nldas_out_dir / f'{YYYY}_{MM:02}_{DD:02}_{HH:02}.nc'
    if nldas_outf_path.is_file():
        print(f'{nldas_outf_path} already exists, not overwriting.')
        pass
        # do not overwrite!
    else:
        #data_url = f'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.002/{YYYY}/{DOY:03}/NLDAS_FORA0125_H.A{YYYY}{MM:02}{DD:02}.{HH:02}00.002.grb'
        data_url = f'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/{YYYY}/{DOY:03}/NLDAS_FORA0125_H.A{YYYY}{MM:02}{DD:02}.{HH:02}00.020.nc'
        session = requests.Session()
        r1 = session.request('get', data_url)
        r = session.get(r1.url, stream=True, auth=(ed_user, ed_pass))

        # write grib file temporarily
        with open(nldas_outf_path, 'wb') as outf:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    outf.write(chunk)

def read_compiled_input(path):
    """
    Check if required input data exists in file and is formatted appropriately.

    Input files should be hourly or finer temporal frequency, drops hours
    without required input data.
    """
    ret = None
    need_vars = {'latitude', 'longitude', 'ET_corr', 'wind_dir', 'u_star', 'sigma_v', 'zm', 'hc', 'd', 'L'}
    # don't parse dates first check if required inputs exist to save processing time
    df = pd.read_csv(path, index_col='date', parse_dates=False)
    cols = df.columns
    check_1 = need_vars.issubset(cols)
    check_2 = len({'u_mean', 'z0'}.intersection(cols)) >= 1  # need one or the other
    # if either test failed then insufficient input data for footprint, abort
    if not check_1 or not check_2:
        return ret
    ret = df
    ret.index = pd.to_datetime(df.index)
    ret = ret.resample('H').mean()
    lat, lon = ret[['latitude', 'longitude']].values[0]
    keep_vars = need_vars.union({'u_mean', 'z0', 'IGBP_land_classification', 'secondary_veg_type'})
    drop_vars = list(set(cols).difference(keep_vars))
    ret.drop(drop_vars, 1, inplace=True)
    ret.dropna(subset=['wind_dir', 'u_star', 'sigma_v', 'd', 'zm', 'L', 'ET_corr'], how='any', inplace=True)
    return ret, lat, lon

def snap_centroid(station_x, station_y):

    # move coord to snap centroid to 30m grid, minimal distortion
    rx = station_x % 15
    if rx > 7.5:
        station_x += (15 - rx)
        # final coords should be odd factors of 15
        if (station_x / 15) % 2 == 0:
            station_x -= 15
    else:
        station_x -= rx
        if (station_x / 15) % 2 == 0:
            station_x += 15

    ry = station_y % 15
    if ry > 7.5:
        print('ry > 7.5')
        station_y += (15 - ry)
        if (station_y / 15) % 2 == 0:
            station_y -= 15
    else:
        print('ry <= 7.5')
        station_y -= ry
        if (station_y / 15) % 2 == 0:
            station_y += 15
    print('adjusted coordinates:', station_x, station_y)
    return station_x, station_y

def extract_nldas_xr_to_df(years,
                           input_path = '.',
                           config_path='../../station_config/',
                           secrets_path="../../secrets/config.ini",
                           output_path="./output/"
                           ):

    if isinstance(input_path, Path):
        pass
    else:
        input_path = Path(input_path)

    if isinstance(config_path, Path):
        pass
    else:
        config_path = Path(config_path)

    if isinstance(output_path, Path):
        pass
    else:
        output_path = Path(output_path)


    dataset = {}
    for year in years:
        ds = xarray.open_dataset(input_path / f"{year}_with_eto.nc", )
        dfs = {}
        for file in config_path.glob('US*.ini'):
            name = file.stem
            print(name)
            config = _load_configs(name, config_path=config_path, secrets_path=secrets_path)

            # Define the target latitude, longitude, and elevation (adjust as needed)
            target_lat = pd.to_numeric(config["latitude"])
            target_lon = pd.to_numeric(config["longitude"])
            target_elev = int(pd.to_numeric(config["elevation"]))

            # Find the nearest latitude, longitude, and elevation in the dataset
            nearest_lat = ds["lat"].sel(lat=target_lat, method="nearest").values
            nearest_lon = ds["lon"].sel(lon=target_lon, method="nearest").values
            nearest_elev = ds["elevation"].sel(elevation=target_elev, method="nearest").values

            # Extract ETo time series at the nearest matching location
            eto_timeseries = ds["ETo"].sel(elevation=nearest_elev, lat=nearest_lat, lon=nearest_lon)
            etr_timeseries = ds["ETr"].sel(elevation=nearest_elev, lat=nearest_lat, lon=nearest_lon)

            # Extract PotEvap time series at the same location
            pet_ts = ds["PotEvap"].sel(lat=nearest_lat, lon=nearest_lon)
            lwd_ts = ds["LWdown"].sel(lat=nearest_lat, lon=nearest_lon)
            swd_ts = ds["SWdown"].sel(lat=nearest_lat, lon=nearest_lon)
            temp_ts = ds["Tair"].sel(lat=nearest_lat, lon=nearest_lon)
            rh_ts = ds["Qair"].sel(lat=nearest_lat, lon=nearest_lon)
            pres_ts = ds['PSurf'].sel(lat=nearest_lat, lon=nearest_lon)
            wind_u_ts = ds['Wind_E'].sel(lat=nearest_lat, lon=nearest_lon)
            wind_v_ts = ds['Wind_N'].sel(lat=nearest_lat, lon=nearest_lon)
            wind_ts = np.sqrt(wind_u_ts ** 2 + wind_v_ts ** 2)

            dfs[name] = pd.DataFrame({'datetime': ds["time"],
                                      'eto': eto_timeseries,
                                      'etr': etr_timeseries,
                                      'pet': pet_ts,
                                      'lwdown': lwd_ts,
                                      'swdown': swd_ts,
                                      'temperature': temp_ts,
                                      'rh': rh_ts,
                                      'pressure': pres_ts,
                                      'wind': wind_ts,
                                      }).round(4)
        dataset[year] = pd.concat(dfs)
    alldata = pd.concat(dataset)
    alldata['datetime'] = pd.to_datetime(alldata['datetime'])
    eto_df = alldata.reset_index().rename(columns={'level_0': 'year',
                                                  'level_1': 'stationid'})
    eto_df['datetime'] = eto_df['datetime'] - pd.Timedelta(hours=7)
    eto_df = eto_df.set_index(['stationid', 'datetime'])

    # Save DataFrame to Parquet
    eto_df.to_parquet(output_path / 'nldas_all.parquet')


def norm_minmax_dly_et(x):
    # Normalize using min-max scaling and then divide by the sum,
    # rounding to 4 decimal places.
    return np.round(((x - x.min()) / (x.max() - x.min())), 4)


def norm_dly_et(x):
    return np.round(x / x.sum(), 4)


def normalize_eto_df(eto_df, eto_field='eto'):
    # Assuming eto_df has a datetime index or a 'datetime' column:
    df = eto_df.reset_index()
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    # Define the desired hour range (inclusive start, exclusive end)
    start_hour = 6
    end_hour = 18

    # Filter the DataFrame for only the hours within the specified range
    mask = (df['hour'] >= start_hour) & (df['hour'] <= end_hour)
    df_subset = df.loc[mask].copy()

    # Apply the normalization for each station and each day
    df_subset['daily_min_max_ETo'] = df_subset.groupby(['stationid', 'date'])[eto_field].transform(norm_minmax_dly_et)
    df_subset['daily_ETo_normed'] = df_subset.groupby(['stationid', 'date'])['daily_min_max_ETo'].transform(norm_dly_et)
    df_subset = df_subset.set_index(['stationid', 'datetime'])

    df_final = pd.merge(eto_df,
                        df_subset[['daily_min_max_ETo', 'daily_ETo_normed']],
                        how='left',
                        left_index=True,
                        right_index=True, )
    # df_final = df.merge(df_subset[['datetime', 'daily_ETo_normed']], on='datetime', how='left')
    df_final['daily_ETo_normed'] = df_final['daily_ETo_normed'].fillna(0)
    return df_final


def calc_nldas_refet(date, hour, nldas_out_dir, latitude, longitude, elevation, zm):
    YYYY = date.year
    DOY = date.timetuple().tm_yday
    MM = date.month
    DD = date.day
    HH = hour
    # already ensured to exist above loop
    nldas_outf_path = nldas_out_dir / f'{YYYY}_{MM:02}_{DD:02}_{HH:02}.grb'
    # open grib and extract needed data at nearest gridcell, calc ETr/ETo anf append to time series
    ds = xarray.open_dataset(nldas_outf_path,engine='pynio').sel(lat_110=latitude, lon_110=longitude, method='nearest')
    # calculate hourly ea from specific humidity
    pair = ds.get('PRES_110_SFC').data / 1000 # nldas air pres in Pa convert to kPa
    sph = ds.get('SPF_H_110_HTGL').data # kg/kg
    ea = refet.calcs._actual_vapor_pressure(q=sph, pair=pair) # ea in kPa
    # calculate hourly wind
    wind_u = ds.get('U_GRD_110_HTGL').data
    wind_v = ds.get('V_GRD_110_HTGL').data
    wind = np.sqrt(wind_u ** 2 + wind_v ** 2)
    # get temp convert to C
    temp = ds.get('TMP_110_HTGL').data - 273.15
    # get rs
    rs = ds.get('DSWRF_110_SFC').data
    unit_dict = {'rs': 'w/m2'}
    # create refet object for calculating

    refet_obj = refet.Hourly(
        tmean=temp, ea=ea, rs=rs, uz=wind,
        zw=zm, elev=elevation, lat=latitude, lon=longitude,
        doy=DOY, time=HH, method='asce', input_units=unit_dict) #HH must be int

    out_dir = Path('All_output')/'AMF'/f'{station}'

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    # this one is saved under the site_ID subdir
    nldas_ts_outf = out_dir/ f'nldas_ETr.csv'
    # save/append time series of point data
    dt = pd.datetime(YYYY,MM,DD,HH)
    ETr_df = pd.DataFrame(columns=['ETr','ETo','ea','sph','wind','pair','temp','rs'])
    ETr_df.loc[dt, 'ETr'] = refet_obj.etr()[0]
    ETr_df.loc[dt, 'ETo'] = refet_obj.eto()[0]
    ETr_df.loc[dt, 'ea'] = ea[0]
    ETr_df.loc[dt, 'sph'] = sph
    ETr_df.loc[dt, 'wind'] = wind
    ETr_df.loc[dt, 'pair'] = pair
    ETr_df.loc[dt, 'temp'] = temp
    ETr_df.loc[dt, 'rs'] = rs
    ETr_df.index.name = 'date'


    # if first run save file with individual datetime (hour data) else open and overwrite hour
    if not nldas_ts_outf.is_file():
        ETr_df.round(4).to_csv(nldas_ts_outf)
        nldas_df = ETr_df.round(4)
    else:
        curr_df = pd.read_csv(nldas_ts_outf, index_col='date', parse_dates=True)
        curr_df.loc[dt] = ETr_df.loc[dt]
        curr_df.round(4).to_csv(nldas_ts_outf)
        nldas_df = curr_df.round(4)

    return nldas_df


class ffp_climatology_new:
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: n.kljun@swansea.ac.uk

    This function calculates footprints within a fixed physical domain for a series of
    time steps, rotates footprints into the corresponding wind direction and aggregates
    all footprints to a footprint climatology. The percentage of source area is
    calculated for the footprint climatology.
    For determining the optimal extent of the domain (large enough to include footprints)
    use calc_footprint_FFP.py.

    FFP Input
        All vectors need to be of equal length (one value for each time step)
        zm       = Measurement height above displacement height (i.e. z-d) [m]
                   usually a scalar, but can also be a vector
        z0       = Roughness length [m] - enter [None] if not known
                   usually a scalar, but can also be a vector
        umean    = Vector of mean wind speed at zm [ms-1] - enter [None] if not known
                   Either z0 or umean is required. If both are given,
                   z0 is selected to calculate the footprint
        h        = Vector of boundary layer height [m]
        ol       = Vector of Obukhov length [m]
        sigmav   = Vector of standard deviation of lateral velocity fluctuations [ms-1]
        ustar    = Vector of friction velocity [ms-1]
        wind_dir = Vector of wind direction in degrees (of 360) for rotation of the footprint

        Optional input:
        domain       = Domain size as an array of [xmin xmax ymin ymax] [m].
                       Footprint will be calculated for a measurement at [0 0 zm] m
                       Default is smallest area including the r% footprint or [-1000 1000 -1000 1000]m,
                       whichever smallest (80% footprint if r not given).
        dx, dy       = Cell size of domain [m]
                       Small dx, dy results in higher spatial resolution and higher computing time
                       Default is dx = dy = 2 m. If only dx is given, dx=dy.
        nx, ny       = Two integer scalars defining the number of grid elements in x and y
                       Large nx/ny result in higher spatial resolution and higher computing time
                       Default is nx = ny = 1000. If only nx is given, nx=ny.
                       If both dx/dy and nx/ny are given, dx/dy is given priority if the domain is also specified.
        rs           = Percentage of source area for which to provide contours, must be between 10% and 90%.
                       Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
                       Expressed either in percentages ("80") or as fractions of 1 ("0.8").
                       Default is [10:10:80]. Set to "None" for no output of percentages
        rslayer      = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
                       Note that this only gives a rough estimate of the footprint as the model is not
                       valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
                       z0 is needed for estimation of the RS.
        smooth_data  = Apply convolution filter to smooth footprint climatology if smooth_data=1 (default)
        crop         = Crop output area to size of the 80% footprint or the largest r given if crop=1
        pulse        = Display progress of footprint calculations every pulse-th footprint (e.g., "100")
        verbosity    = Level of verbosity at run time: 0 = completely silent, 1 = notify only of fatal errors,
                       2 = all notifications
        fig          = Plot an example figure of the resulting footprint (on the screen): set fig = 1.
                       Default is 0 (i.e. no figure).

    FFP output
        FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
        x_2d	    = x-grid of 2-dimensional footprint [m]
        y_2d	    = y-grid of 2-dimensional footprint [m]
        fclim_2d = Normalised footprint function values of footprint climatology [m-2]
        rs       = Percentage of footprint as in input, if provided
        fr       = Footprint value at r, if r is provided
        xr       = x-array for contour line of r, if r is provided
        yr       = y-array for contour line of r, if r is provided
        n        = Number of footprints calculated and included in footprint climatology
        flag_err = 0 if no error, 1 in case of error, 2 if not all contour plots (rs%) within specified domain,
                   3 if single data points had to be removed (outside validity)

    Created: 19 May 2016 natascha kljun
    Converted from matlab to python, together with Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.4
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015,2016,2017,2018,2019,2020 Natascha Kljun
    """

    def __init__(
        self,
        df: pd.DataFrame,
        domain: np.ndarray = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: int = 2,
        dy: int = 2,
        nx: int = 1000,
        ny: int = 1000,
        rs: Union[list, np.ndarray] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        crop_height: float = 0.2,
        atm_bound_height: float = 2000.0,
        inst_height: float = 2.0,
        rslayer: bool = False,
        smooth_data: bool = True,
        crop: bool = False,
        verbosity: int = 2,
        fig: bool = False,
        logger=None,
        time=None,
        crs: int = None,
        station_x: float = None,
        station_y: float = None,
        **kwargs,
    ):

        self.df = df

        # Model parameters
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.x = np.linspace(self.xmin, self.xmax, nx + 1)
        self.y = np.linspace(self.ymin, self.ymax, ny + 1)

        self.rotated_theta = None

        self.a = 1.4524
        self.b = -1.9914
        self.c = 1.4622
        self.d = 0.1359
        self.ac = 2.17
        self.bc = 1.66
        self.cc = 20.0
        self.oln = 5000.0  # limit to L for neutral scaling
        self.k = 0.4  # von Karman
        # ===========================================================================
        # Get kwargs
        self.show_heatmap = kwargs.get("show_heatmap", True)

        # ===========================================================================
        # Input check
        self.flag_err = 0

        self.dx = dx
        self.dy = dy

        self.rs = rs
        self.rslayer = rslayer
        self.smooth_data = smooth_data
        self.crop = crop
        self.verbosity = verbosity
        self.fig = fig

        self.time = None  # defined later after dropping na values
        self.ts_len = None  # defined by len of dropped df

        self.f_2d = None

        self.logger = logger

        if self.verbosity < 2:
            self.logger.setLevel(logging.INFO)
        elif self.verbosity < 3:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.DEBUG)

        if "crop_height" in df.columns:
            h_c = df["crop_height"]
        else:
            h_c = crop_height

        if "atm_bound_height" in df.columns:
            h_s = df["atm_bound_height"]
        else:
            h_s = atm_bound_height

        if "inst_height" in df.columns:
            zm_s = df["inst_height"]
        else:
            zm_s = inst_height

        self.prep_df_fields(
            h_c=h_c,
            d_h=None,
            zm_s=zm_s,
            h_s=h_s,
        )
        self.define_domain()
        self.create_xr_dataset()

    def prep_df_fields(
        self,
        h_c=0.2,
        d_h=None,
        zm_s=2.0,
        h_s=2000.0,
    ):
        # h_c Height of canopy [m]
        # Estimated displacement height [m]
        # zm_s Measurement height [m] from AMF metadata
        # h_s Height of atmos. boundary layer [m] - assumed

        if d_h is None:
            d_h = 10 ** (0.979 * np.log10(h_c) - 0.154)

        self.df["zm"] = zm_s - d_h
        self.df["h_c"] = h_c
        self.df["z0"] = h_c * 0.123
        self.df["h"] = h_s

        self.df = self.df.rename(
            columns={
                "V_SIGMA": "sigmav",
                "USTAR": "ustar",
                "wd": "wind_dir",
                "MO_LENGTH": "ol",
                "ws": "umean",
            }
        )

        self.df["zm"] = np.where(self.df["zm"] <= 0.0, np.nan, self.df["zm"])
        self.df["h"] = np.where(self.df["h"] <= 10.0, np.nan, self.df["h"])
        self.df["zm"] = np.where(self.df["zm"] > self.df["h"], np.nan, self.df["zm"])
        self.df["sigmav"] = np.where(self.df["sigmav"] < 0.0, np.nan, self.df["sigmav"])
        self.df["ustar"] = np.where(self.df["ustar"] <= 0.1, np.nan, self.df["ustar"])

        self.df["wind_dir"] = np.where(
            self.df["wind_dir"] > 360.0, np.nan, self.df["wind_dir"]
        )
        self.df["wind_dir"] = np.where(
            self.df["wind_dir"] < 0.0, np.nan, self.df["wind_dir"]
        )

        # Check if all required fields are in the DataFrame
        all_present = np.all(
            np.isin(["ol", "sigmav", "ustar", "wind_dir"], self.df.columns)
        )
        if all_present:
            self.logger.debug("All required fields are present")
        else:
            self.raise_ffp_exception(1)

        self.df = self.df.dropna(subset=["sigmav", "wind_dir", "h", "ol"], how="any")
        self.ts_len = len(self.df)
        self.logger.debug(f"input len is {self.ts_len}")

    def raise_ffp_exception(self, code):
        exceptions = {
            1: "At least one required parameter is missing. Check the inputs.",
            2: "zm (measurement height) must be larger than zero.",
            3: "z0 (roughness length) must be larger than zero.",
            4: "h (boundary layer height) must be larger than 10 m.",
            5: "zm (measurement height) must be smaller than h (boundary layer height).",
            6: "zm (measurement height) should be above the roughness sub-layer.",
            7: "zm/ol (measurement height to Obukhov length ratio) must be >= -15.5.",
            8: "sigmav (standard deviation of crosswind) must be larger than zero.",
            9: "ustar (friction velocity) must be >= 0.1.",
            10: "wind_dir (wind direction) must be in the range [0, 360].",
        }

        message = exceptions.get(code, "Unknown error code.")

        if self.verbosity > 0:
            print(f"Error {code}: {message}")
            self.logger.info(f"Error {code}: {message}")

        if code in [1, 4, 5, 7, 9, 10]:  # Fatal errors
            self.logger.error(f"Error {code}: {message}")
            raise ValueError(f"FFP Exception {code}: {message}")

    def define_domain(self):
        # ===========================================================================
        # Create 2D grid
        self.xv, self.yv = np.meshgrid(self.x, self.y, indexing="xy")

        # Define physical domain in cartesian and polar coordinates
        self.logger.debug(f"x: {self.x}, y: {self.y}")
        # Polar coordinates
        # Set theta such that North is pointing upwards and angles increase clockwise
        # Polar coordinates
        self.rho = xarray.DataArray(
            np.sqrt(self.xv**2 + self.yv**2),
            dims=("x", "y"),
            coords={"x": self.x, "y": self.y},
        )
        self.theta = xarray.DataArray(
            np.arctan2(self.yv, self.xv),
            dims=("x", "y"),
            coords={"x": self.x, "y": self.y},
        )
        self.logger.debug(f"rho: {self.rho}, theta: {self.theta}")
        # Initialize raster for footprint climatology
        self.fclim_2d = xarray.zeros_like(self.rho)

        # ===========================================================================

    def create_xr_dataset(self):
        # Time series inputs as an xarray.Dataset
        self.df.index.name = "time"
        self.ds = xarray.Dataset.from_dataframe(self.df)

    def calc_xr_footprint(self):
        # Rotate coordinates into wind direction
        self.rotated_theta = self.theta - (self.ds["wind_dir"] * np.pi / 180.0)

        psi_cond = np.logical_and(self.oln > self.ds["ol"], self.ds["ol"] > 0)

        # Compute xstar_ci_dummy for all timestamps
        xx = (1.0 - 19.0 * self.ds["zm"] / self.ds["ol"]) ** 0.25

        psi_f = xarray.where(
            psi_cond,
            -5.3 * self.ds["zm"] / self.ds["ol"],
            np.log((1.0 + xx**2) / 2.0)
            + 2.0 * np.log((1.0 + xx) / 2.0)
            - 2.0 * np.arctan(xx)
            + np.pi / 2.0,
        )

        xstar_bottom = xarray.where(
            self.ds["z0"].isnull(),
            (self.ds["umean"] / self.ds["ustar"] * self.k),
            (np.log(self.ds["zm"] / self.ds["z0"]) - psi_f),
        )

        xstar_ci_dummy = xarray.where(
            (np.log(self.ds["zm"] / self.ds["z0"]) - psi_f) > 0,
            self.rho
            * np.cos(self.rotated_theta)
            / self.ds["zm"]
            * (1.0 - (self.ds["zm"] / self.ds["h"]))
            / xstar_bottom,
            0.0,
        )

        xstar_ci_dummy = xstar_ci_dummy.astype(float)
        # Mask invalid values
        px = xstar_ci_dummy > self.d

        # Compute fstar_ci_dummy and f_ci_dummy
        fstar_ci_dummy = xarray.where(
            px,
            self.a
            * (xstar_ci_dummy - self.d) ** self.b
            * np.exp(-self.c / (xstar_ci_dummy - self.d)),
            0.0,
        )

        f_ci_dummy = xarray.where(
            px,
            fstar_ci_dummy
            / self.ds["zm"]
            * (1.0 - (self.ds["zm"] / self.ds["h"]))
            / xstar_bottom,
            0.0,
        )

        # Calculate sigystar_dummy for valid points
        sigystar_dummy = xarray.where(
            px,
            self.ac
            * np.sqrt(
                self.bc
                * np.abs(xstar_ci_dummy) ** 2
                / (1.0 + self.cc * np.abs(xstar_ci_dummy))
            ),
            0.0,  # Default value for invalid points
        )

        self.ds["ol"] = xarray.where(self.ds["ol"] > self.oln, -1e6, self.ds["ol"])

        # Calculate scale_const in a vectorized way
        scale_const = xarray.where(
            self.ds["ol"] <= 0,
            1e-5 * abs(self.ds["zm"] / self.ds["ol"]) ** (-1) + 0.80,
            1e-5 * abs(self.ds["zm"] / self.ds["ol"]) ** (-1) + 0.55,
        )
        scale_const = xarray.where(scale_const > 1.0, 1.0, scale_const)

        # Calculate sigy_dummy
        sigy_dummy = xarray.where(
            px,
            sigystar_dummy
            / scale_const
            * self.ds["zm"]
            * self.ds["sigmav"]
            / self.ds["ustar"],
            0.0,  # Default value for invalid points
        )

        sigy_dummy = xarray.where(sigy_dummy <= 0.0, np.nan, sigy_dummy)

        # sig_cond = np.logical_or(sigy_dummy.isnull(), px, sigy_dummy == 0.0)

        # Calculate the footprint in real scale
        self.f_2d = xarray.where(
            sigy_dummy.isnull(),
            0.0,
            f_ci_dummy
            / (np.sqrt(2 * np.pi) * sigy_dummy)
            * np.exp(
                -((self.rho * np.sin(self.rotated_theta)) ** 2) / (2.0 * sigy_dummy**2)
            ),
        )

        # self.f_2d = xr.where(px, self.f_2d, 0.0)

        # Accumulate into footprint climatology raster
        self.fclim_2d = self.f_2d.sum(dim="time")

        # Apply smoothing if requested
        if self.smooth_data:
            self.f_2d = xarray.apply_ufunc(
                gaussian_filter,
                self.f_2d,
                kwargs={"sigma": 1.0},
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]],
            )

    def smooth_and_contour(self, rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        """
        Compute footprint climatology using xarray structures for efficient, vectorized operations.

        Parameters:
            rs (list): Contour levels to compute.
            smooth_data (bool): Whether to smooth data using Gaussian filtering.

        Returns:
            xr.Dataset: Aggregated footprint climatology.
        """

        # Ensure the footprint data is normalized
        self.ds["footprint"] = self.ds["footprint"] / self.ds["footprint"].sum(
            dim=("x", "y")
        )

        # Apply smoothing if requested
        if self.smooth_data:
            self.ds["footprint"] = xarray.apply_ufunc(
                gaussian_filter,
                self.ds["footprint"],
                kwargs={"sigma": 1.0},
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]],
            )

        # Calculate cumulative footprint and extract contours
        cumulative = self.ds["footprint"].cumsum(dim="x").cumsum(dim="y")

        contours = {r: cumulative.where(cumulative >= r).fillna(0) for r in self.rs}

        # Combine results into a dataset
        climatology = xarray.Dataset(
            {f"contour_{int(r * 100)}": data for r, data in contours.items()}
        )

        return climatology

    def run(self):
        self.calc_xr_footprint()
        # self.smooth_and_contour()
        return {
            "x_2d": self.xv,
            "y_2d": self.yv,
            "fclim_2d": self.fclim_2d,
            "f_2d": self.f_2d,
            "rs": self.rs,
        }


if __name__ == "__main__":

    # load initial flux data
    d = Data("US-CRT_config.ini")
    # adding variable names to Data instance name list for resampling
    d.variables["MO_LENGTH"] = "MO_LENGTH"
    d.variables["USTAR"] = "USTAR"
    d.variables["V_SIGMA"] = "V_SIGMA"
    # renaming columns (optional and only affects windspeed and wind direction names)
    df = d.df.rename(columns=d.inv_map)
    df = df.resample("h").mean()
    # get coords info from Data instance
    latitude = d.latitude
    longitude = d.longitude
    station_coord = (latitude, longitude)
    station = d.site_id
    # get EPSG code from lat,long, convert to UTM https://epsg.io/32617
    EPSG = int(
        32700
        - np.round((45 + latitude) / 90.0) * 100
        + np.round((183 + longitude) / 6.0)
    )
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{int(EPSG)}")
    (station_x, station_y) = transformer.transform(*station_coord)
    # check results, EPSG code should be 32617, lon should be near 304485 and lat 4611191
    h_c = 0.2  # Height of canopy [m]
    # Estimated displacement height [m]
    d = 10 ** (0.979 * np.log10(h_c) - 0.154)
    # Other model parameters
    zm_s = 2.0  # Measurement height [m] from AMF metadata
    h_s = 2000.0  # Height of atmos. boundary layer [m] - assumed
    dx = 3.0  # Model resolution [m]
    origin_d = 200.0  # Model bounds distance from origin [m]
    # from 7 AM to 8 PM only, modify if needed
    start_hr = 7
    end_hr = 20
    hours = np.arange(start_hr, end_hr + 1)

    # Loop through each day in the dataframe
    for date in df.index.date:

        # Subset dataframe to only values in day of year
        print(f"Date: {date}")
        temp_df = df[df.index.date == date]

        new_dat = None

        for indx, t in enumerate(hours):

            band = indx + 1
            print(f"Hour: {t}")

            try:
                temp_line = temp_df.loc[temp_df.index.hour == t, :]

                # Calculate footprint
                temp_ffp = ffp_climatology(
                    domain=[-origin_d, origin_d, -origin_d, origin_d],
                    dx=dx,
                    dy=dx,
                    zm=zm_s - d,
                    h=h_s,
                    rs=None,
                    z0=h_c * 0.123,
                    ol=temp_line["MO_LENGTH"].values,
                    sigmav=temp_line["V_SIGMA"].values,
                    ustar=temp_line["USTAR"].values,  # umean=temp_line['ws'].values,
                    wind_dir=temp_line["wd"].values,
                    crop=0,
                    fig=0,
                    verbosity=0,
                )
                ####verbosoity=2 prints out errors; if z0 triggers errors, use umean
                #    print(zm_s-d)

                f_2d = np.array(temp_ffp["fclim_2d"])
                x_2d = np.array(temp_ffp["x_2d"]) + station_x
                y_2d = np.array(temp_ffp["y_2d"]) + station_y
                f_2d = f_2d * dx**2

                # Calculate affine transform for given x_2d and y_2d
                affine_transform = find_transform(x_2d, y_2d)

                # Create data file if not already created
                if new_dat is None:
                    out_f = f"./{date}_{station}.tif"
                    print(f_2d.shape)
                    new_dat = rasterio.open(
                        out_f,
                        "w",
                        driver="GTiff",
                        dtype=rasterio.float64,
                        count=len(hours),
                        height=f_2d.shape[0],
                        width=f_2d.shape[1],
                        transform=affine_transform,
                        crs=pyproj.crs.CRS.from_epsg(int(EPSG)),
                        nodata=0.00000000e000,
                    )

            except Exception as e:

                print(f"Hour {t} footprint failed, band {band} not written.")

                temp_ffp = None

                continue

            # Mask out points that are below a % threshold (defaults to 90%)
            f_2d = mask_fp_cutoff(f_2d)

            # Write the new band
            new_dat.write(f_2d, indx + 1)

            # Update tags with metadata
            tag_dict = {
                "hour": f"{t * 100:04}",
                "wind_dir": temp_line["wd"].values,
                "total_footprint": np.nansum(f_2d),
            }

            new_dat.update_tags(indx + 1, **tag_dict)

        # Close dataset if it exists
        try:
            new_dat.close()
        except:
            continue

        print()

        # for esample just create a single day and exit
        break
