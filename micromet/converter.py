# Written By Kat Ladig and Paul Inkenbrandt
from pathlib import Path
from typing import Union, List, Dict, Optional

import yaml

import datetime
import pathlib
import os
import numpy as np
import pandas as pd
import csv

from .outlier_removal import replace_flat_values

import logging

# Set up a logger for this module.
# In production, consider configuring logging at application start-up instead.
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class AmerifluxDataProcessor:
    NA_VALUES = {"-9999", "NAN", "NaN", "nan", np.nan, -9999.0}

    def __init__(self):
        self.headers = {
            "default": [
                "TIMESTAMP_START",
                "TIMESTAMP_END",
                "CO2",
                "CO2_SIGMA",
                "H2O",
                "H2O_SIGMA",
                "FC",
                "FC_SSITC_TEST",
                "LE",
                "LE_SSITC_TEST",
                "ET",
                "ET_SSITC_TEST",
                "H",
                "H_SSITC_TEST",
                "G",
                "SG",
                "FETCH_MAX",
                "FETCH_90",
                "FETCH_55",
                "FETCH_40",
                "WD",
                "WS",
                "WS_MAX",
                "USTAR",
                "ZL",
                "TAU",
                "TAU_SSITC_TEST",
                "MO_LENGTH",
                "U",
                "U_SIGMA",
                "V",
                "V_SIGMA",
                "W",
                "W_SIGMA",
                "PA",
                "TA_1_1_1",
                "RH_1_1_1",
                "T_DP_1_1_1",
                "TA_1_1_2",
                "RH_1_1_2",
                "T_DP_1_1_2",
                "TA_1_1_3",
                "RH_1_1_3",
                "T_DP_1_1_3",
                "TA_1_1_4",
                "VPD",
                "T_SONIC",
                "T_SONIC_SIGMA",
                "PBLH",
                "TS_1_1_1",
                "TS_1_1_2",
                "SWC_1_1_1",
                "SWC_1_1_2",
                "ALB",
                "NETRAD",
                "SW_IN",
                "SW_OUT",
                "LW_IN",
                "LW_OUT",
                "P",
            ],
            "math_soils": [
                "VWC_5cm_N_Avg",
                "VWC_5cm_S_Avg",
                "Ka_5cm_N_Avg",
                "T_5cm_N_Avg",
                "BulkEC_5cm_N_Avg",
                "VWC_10cm_N_Avg",
                "Ka_10cm_N_Avg",
                "T_10cm_N_Avg",
                "BulkEC_10cm_N_Avg",
                "VWC_20cm_N_Avg",
                "Ka_20cm_N_Avg",
                "T_20cm_N_Avg",
                "BulkEC_20cm_N_Avg",
                "VWC_30cm_N_Avg",
                "Ka_30cm_N_Avg",
                "T_30cm_N_Avg",
                "BulkEC_30cm_N_Avg",
                "VWC_40cm_N_Avg",
                "Ka_40cm_N_Avg",
                "T_40cm_N_Avg",
                "BulkEC_40cm_N_Avg",
                "VWC_50cm_N_Avg",
                "Ka_50cm_N_Avg",
                "T_50cm_N_Avg",
                "BulkEC_50cm_N_Avg",
                "VWC_60cm_N_Avg",
                "Ka_60cm_N_Avg",
                "T_60cm_N_Avg",
                "BulkEC_60cm_N_Avg",
                "VWC_75cm_N_Avg",
                "Ka_75cm_N_Avg",
                "T_75cm_N_Avg",
                "BulkEC_75cm_N_Avg",
                "VWC_100cm_N_Avg",
                "Ka_100cm_N_Avg",
                "T_100cm_N_Avg",
                "BulkEC_100cm_N_Avg",
                "Ka_5cm_S_Avg",
                "T_5cm_S_Avg",
                "BulkEC_5cm_S_Avg",
                "VWC_10cm_S_Avg",
                "Ka_10cm_S_Avg",
                "T_10cm_S_Avg",
                "BulkEC_10cm_S_Avg",
                "VWC_20cm_S_Avg",
                "Ka_20cm_S_Avg",
                "T_20cm_S_Avg",
                "BulkEC_20cm_S_Avg",
                "VWC_30cm_S_Avg",
                "Ka_30cm_S_Avg",
                "T_30cm_S_Avg",
                "BulkEC_30cm_S_Avg",
                "VWC_40cm_S_Avg",
                "Ka_40cm_S_Avg",
                "T_40cm_S_Avg",
                "BulkEC_40cm_S_Avg",
                "VWC_50cm_S_Avg",
                "Ka_50cm_S_Avg",
                "T_50cm_S_Avg",
                "BulkEC_50cm_S_Avg",
                "VWC_60cm_S_Avg",
                "Ka_60cm_S_Avg",
                "T_60cm_S_Avg",
                "BulkEC_60cm_S_Avg",
                "VWC_75cm_S_Avg",
                "Ka_75cm_S_Avg",
                "T_75cm_S_Avg",
                "BulkEC_75cm_S_Avg",
                "VWC_100cm_S_Avg",
                "Ka_100cm_S_Avg",
                "T_100cm_S_Avg",
            ],
            "well_soils": [
                "SWC_3_1_1",
                "SWC_4_1_1",
                "K_3_1_1",
                "TS_3_1_1",
                "EC_3_1_1",
                "SWC_3_2_1",
                "K_3_2_1",
                "TS_3_2_1",
                "EC_3_2_1",
                "SWC_3_3_1",
                "K_3_3_1",
                "TS_3_3_1",
                "EC_3_3_1",
                "SWC_3_4_1",
                "K_3_4_1",
                "TS_3_4_1",
                "EC_3_4_1",
                "SWC_3_5_1",
                "K_3_5_1",
                "TS_3_5_1",
                "EC_3_5_1",
                "SWC_3_6_1",
                "K_3_6_1",
                "TS_3_6_1",
                "EC_3_6_1",
                "SWC_3_7_1",
                "K_3_7_1",
                "TS_3_7_1",
                "EC_3_7_1",
                "SWC_3_8_1",
                "K_3_8_1",
                "TS_3_8_1",
                "EC_3_8_1",
                "SWC_3_9_1",
                "K_3_9_1",
                "TS_3_9_1",
                "EC_3_9_1",
            ],
            "math_soils_v2": [
                "SWC_3_1_1",
                "SWC_4_1_1",
                "K_3_1_1",
                "TS_3_1_1",
                "EC_3_1_1",
                "SWC_3_2_1",
                "K_3_2_1",
                "TS_3_2_1",
                "EC_3_2_1",
                "SWC_3_3_1",
                "K_3_3_1",
                "TS_3_3_1",
                "EC_3_3_1",
                "SWC_3_4_1",
                "K_3_4_1",
                "TS_3_4_1",
                "EC_3_4_1",
                "SWC_3_5_1",
                "K_3_5_1",
                "TS_3_5_1",
                "EC_3_5_1",
                "SWC_3_6_1",
                "K_3_6_1",
                "TS_3_6_1",
                "EC_3_6_1",
                "SWC_3_7_1",
                "K_3_7_1",
                "TS_3_7_1",
                "EC_3_7_1",
                "SWC_3_8_1",
                "K_3_8_1",
                "TS_3_8_1",
                "EC_3_8_1",
                "SWC_3_9_1",
                "K_3_9_1",
                "TS_3_9_1",
                "EC_3_9_1",
                "K_4_1_1",
                "TS_4_1_1",
                "EC_4_1_1",
                "SWC_4_2_1",
                "K_4_2_1",
                "TS_4_2_1",
                "EC_4_2_1",
                "SWC_4_3_1",
                "K_4_3_1",
                "TS_4_3_1",
                "EC_4_3_1",
                "SWC_4_4_1",
                "K_4_4_1",
                "TS_4_4_1",
                "EC_4_4_1",
                "SWC_4_5_1",
                "K_4_5_1",
                "TS_4_5_1",
                "EC_4_5_1",
                "SWC_4_6_1",
                "K_4_6_1",
                "TS_4_6_1",
                "EC_4_6_1",
                "SWC_4_7_1",
                "K_4_7_1",
                "TS_4_7_1",
                "EC_4_7_1",
                "SWC_4_8_1",
                "K_4_8_1",
                "TS_4_8_1",
                "EC_4_8_1",
                "EC_4_9_1",
                "SWC_4_9_1",
                "K_4_9_1",
                "TS_4_9_1",
                "TS_1_1_1",
                "TS_2_1_1",
                "SWC_1_1_1",
                "SWC_2_1_1",
            ],
        }

        # Simplify derived headers
        self.headers["bflat"] = [
            h
            for h in self.headers["default"]
            if h not in {"TA_1_1_4", "TS_1_1_2", "SWC_1_1_2"}
        ]
        self.headers["wellington"] = [
            h for h in self.headers["default"] if h != "TS_1_1_1"
        ]

        self.headers["big_math"] = (
            self.headers["wellington"][:-10]
            + self.headers["math_soils"]
            + self.headers["wellington"][-10:]
            + ["T_CANOPY"]
        )
        self.headers["big_math_v2"] = (
            self.headers["wellington"][:-10]
            + self.headers["math_soils_v2"]
            + self.headers["wellington"][-7:]
            + ["T_CANOPY"]
        )
        self.headers["big_math_v2_filt"] = [
            h for h in self.headers["big_math_v2"] if h not in {"TA_1_1_4"}
        ]

        self.headers["big_well"] = [
            h for h in self.headers["default"] if h not in {"TA_1_1_4"}
        ] + self.headers["well_soils"]

        self.header_dict = {
            60: self.headers["default"],
            61: self.headers["bflat"],
            62: self.headers["wellington"],
            96: self.headers["big_well"],
            131: self.headers["big_math"],
            132: self.headers["big_math_v2_filt"],
        }

    @staticmethod
    def check_header(csv_file: Union[str, Path]) -> int:
        """
        Check if the given CSV file has a header row.

        :param csv_file: Path to the CSV file.
        :type csv_file: str
        :return:
            1 if the header contains "TIMESTAMP_START",
            2 if the header contains "TOA5",
            0 otherwise.
        :rtype: int
        """
        try:
            with open(csv_file, "r", newline="", encoding="utf-8") as file:
                first_row = next(csv.reader(file), [])
                if "TIMESTAMP_START" in first_row:
                    return 1
                if "TOA5" in first_row:
                    return 2
                return 0
        except Exception:
            return 0

    @staticmethod
    def check_header(csv_file):
        """
        Check if the given CSV file has a header row.

        :param csv_file: Path to the CSV file.
        :type csv_file: str
        :return:
            1 if the header contains "TIMESTAMP_START",
            2 if the header contains "TOA5",
            0 otherwise.
        :rtype: int
        """

        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                first_row = next(reader)
            except StopIteration:
                # Empty file
                return 0
            if "TIMESTAMP_START" in first_row:
                return 1
            elif "TOA5" in first_row:
                return 2
            else:
                return 0

    def dataframe_from_file(self, file: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Reads a CSV file and returns a DataFrame with appropriate headers based on the file format.

        Parameters:
        - file (str or Path): Path to the CSV file.

        Returns:
        - pd.DataFrame or None: DataFrame with proper headers, or None if unsupported format or errors occur.
        """

        try:
            header_type = self.check_header(file)

            if header_type == 1:
                # Known header with "TIMESTAMP_START"
                return pd.read_csv(file, na_values=self.na_values)

            elif header_type == 2:
                # "TOA5" format, skip known metadata lines
                df = pd.read_csv(file, na_values=self.na_values, skiprows=[0, 2, 3])
                df.drop(columns=["TIMESTAMP"], errors="ignore", inplace=True)

                return df

            else:

                col_count = pd.read_csv(file, header=None, nrows=1).shape[1]
                header = self.header_dict.get(col_count)

                if header:
                    return pd.read_csv(file, names=header, na_values=self.NA_VALUES)

                logger.warning(
                    f"Unknown header format ({col_count} columns) in file: {file}"
                )
                return None

        except pd.errors.EmptyDataError:
            logger.warning(f"No data found in file: {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
        return None

    def _is_int(self, element: any) -> bool:
        """
        Check if the provided element can be cast as an integer.

        Parameters:
        - element (any): The value to check.

        Returns:
        - bool: True if the element can be cast to an integer, False otherwise.
        """
        try:
            return element is not None and float(element).is_integer()
        except (ValueError, TypeError):
            return False

    def raw_file_compile(
        self,
        raw_fold: Path,
        station_folder_name: Union[str, Path],
        search_str="*Flux_AmeriFluxFormat*.dat",
    ) -> Optional[pd.DataFrame]:
        """
        Compiles raw AmeriFlux datalogger files into a single dataframe.

        :param raw_fold: Path to the root folder of raw datalogger files
        :type raw_fold: pathlib.Path
        :param station_folder_name: Name of the station folder containing the raw datalogger files
        :type station_folder_name: str
        :return: Dataframe containing compiled AmeriFlux data, or None if no valid files found
        :rtype: pandas.DataFrame or None
        """
        compiled_data = []
        station_folder = raw_fold / station_folder_name

        logger.info(f"Compiling data from {station_folder}")

        for file in station_folder.rglob(search_str):
            logger.info(f"Processing file: {file}")
            basename = file.stem

            try:
                file_number = int(basename.split("_")[-1])
                datalogger_number = int(basename.split("_")[0])
            except ValueError:
                file_number = datalogger_number = 9999

            df = self.dataframe_from_file(file)
            if df is not None:
                df["file_no"] = file_number
                df["datalogger_no"] = datalogger_number
                compiled_data.append(df)

        if compiled_data:
            return pd.concat(compiled_data, ignore_index=True)
        else:
            logger.warning(f"No valid files found in {station_folder}")
            return None


class Reformatter(object):
    """
    A class for reformatting raw Utah Flux Network data into the AmeriFlux-compatible format.

    Attributes:
        et_data (pd.DataFrame): Processed data after initialization and preparation.
        config (dict): Configuration loaded from YAML file.
        varlimits (pd.DataFrame): Limits for variables loaded from CSV.

    Methods:
        __init__: Initializes Reformatter with data and configuration.
        load_variable_limits: Loads variable limits from file.
        clean_columns: Cleans and processes DataFrame columns, handling missing and invalid values.
        prepare_et_data: Prepares and preprocesses the ET data.
        remove_extra_soil_params: Removes unnecessary soil parameter columns.
        drop_extras: Drops extra columns based on configuration.
        col_order: Reorders columns prioritizing timestamps.
        datefixer: Corrects date and time formats, removes duplicates, and resamples data.
        update_dataframe_column: Updates DataFrame by replacing an old column with a new combined column.
        rename_columns: Renames DataFrame columns based on configuration.
        scale_and_convert: Scales and converts a column to float type.
        ssitc_scale: Scales quality control SSITC columns to a 0-2 scale.
        rating: Categorizes numeric values into ratings.
        _extract_variable_name: Extracts the base variable name for processing.
        replace_out_of_range_with_nan: Replaces out-of-range values with NaN.
        extreme_limiter: Applies limits to all columns to remove extreme values.
        despike: Removes data spikes based on a standard deviation threshold.
        tau_fixer: Corrects sign of TAU variable.
        fix_swc_percent: Converts soil water content to percentage format if necessary.
        timestamp_reset: Resets timestamp columns based on DataFrame index.
        despike_ewma_fb: Removes spikes using forward-backward EWMA smoothing.
    """

    def __init__(
        self,
        et_data,
        config_path="./data/reformatter_vars.yml",
        drop_soil=True,
        data_path=None,
        data_type="eddy",
        spike_threshold=4.5,
        outlier_remove=True,
    ):
        """
        Initializes the Reformatter with ET data and configurations.

        Args:
            et_data (pd.DataFrame): Raw data to process.
            config_path (str): Path to the YAML configuration file.
            drop_soil (bool): Whether to drop unnecessary soil parameters.
            data_path (str): Optional path to the extreme values CSV file.
            data_type (str): Type of dataset, either "eddy" or "met".
            spike_threshold (float): Threshold for spike removal.
            outlier_remove (bool): Whether to remove outliers.
        """
        self.default_paths = [
            pathlib.Path("../data/extreme_values.csv"),
            pathlib.Path("data/extreme_values.csv"),
            pathlib.Path("../../data/extreme_values.csv"),
            pathlib.Path("../../../data/extreme_values.csv"),
            pathlib.Path(
                "G:/Shared drives/Data_Processing/Jupyter_Notebooks/Micromet/data/extreme_values.csv"
            ),
        ]

        self.config = self._load_config(config_path)

        self.data_path = data_path
        self.spike_threshold = spike_threshold
        self.COL_NAME_MATCH = self.config["col_name_match"]
        self.MET_RENAMES = self.config["met_renames"]
        self.MET_VARS = self.config["met_vars"]
        self.DESPIKEY = self.config["despikey"]
        self.DROP_COLS = self.config["drop_cols"]
        self.OTHER_VARS = self.config["othervar"]
        self.MATH_SOILS_V2 = self.config["math_soils_v2"]
        self.MATH_SOILS_V2 = self.config["math_soils_v2"]
        self.DESPIKEY = self.config["despikey"]
        self.varlimits = None
        self.load_variable_limits()
        self.prepare_et_data(et_data, data_type, drop_soil)
        self.prepare_et_data(et_data, data_type, drop_soil)

    @staticmethod
    def _load_config(config_path):

        path = pathlib.Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path.resolve()}")
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def load_variable_limits(self):

        paths_to_try = (
            [pathlib.Path(self.data_path)] if self.data_path else self.default_paths
        )
        for path in paths_to_try:
            if path.exists():
                self.varlimits = pd.read_csv(path, index_col="Name")
                logger.info(f"Loaded variable limits from {path}")
                return
        raise FileNotFoundError(
            "Could not locate extreme_values.csv in provided paths."
        )

    def clean_columns(self):
        """Cleans and preprocesses columns, handling invalid values and despiking."""
        for col in self.et_data.columns:
            logger.warning(f"column: {col}")

            if col in ["MO_LENGTH", "RECORD"]:
                self.et_data[col] = pd.to_numeric(
                    self.et_data[col], downcast="integer", errors="coerce"
                )

            elif col in ["TIMESTAMP_START", "TIMESTAMP_END"]:
                self.et_data[col] = self.et_data[col]

            elif "SSITC" in col:
                self.et_data[col] = pd.to_numeric(
                    self.et_data[col], downcast="integer", errors="coerce"
                )
            else:
                self.et_data[col] = pd.to_numeric(self.et_data[col], errors="coerce")

            logger.debug(f"column {col} range: {np.max(self.et_data[col])}")
            logger.debug(f"column {col} range numeric: {np.max(self.et_data[col])}")

            self.et_data[col] = self.et_data[col].replace(-9999, np.nan)

            # remove values that are outside of possible ranges
            self.et_data = self.replace_out_of_range_with_nan(self.et_data, col, np.nan)

            logger.warning(f"column range out of range: {np.max(self.et_data[col])}")

            if col in self.DESPIKEY:

                # despike
                self.et_data[col] = self.despike(self.et_data[col])

                # Remove Flat Values
                if col in ["U", "V", "W", "u", "w", "v"]:
                    pass
                else:
                    self.et_data[col] = replace_flat_values(
                        self.et_data, col, replacement_value=np.nan, null_value=-9999
                    )
            logger.debug(f"column range despike: {np.max(self.et_data[col])}")

    def prepare_et_data(self, et_data, data_type="eddy", drop_soil=True):
        """
        Prepares ET data by correcting timestamps, renaming columns, and cleaning data.

        Args:
            et_data (pd.DataFrame): Raw ET data.
            data_type (str): Dataset type ("eddy" or "met").
            drop_soil (bool): Whether to drop excess soil parameter columns.
        """
        logger.debug("Starting Processing")
        logger.debug(f"Reading first line of file: {self.data_path}")
        logger.debug(f"Variable limits: \n {self.varlimits.head(5)}")
        logger.debug(f"Variable limits: \n {self.varlimits.tail(5)}")
        logger.debug(f"ET Data: \n {et_data.head(5)}")
        logger.debug(f"ET Data: \n {et_data.tail(5)}")

        # fix datetimes
        self.et_data = self.datefixer(et_data)

        logger.debug("Corrected Datetimes:")
        logger.debug(f"{self.et_data.head(5)}")
        logger.debug(f"{self.et_data.tail(5)}")

        # change variable names
        self.rename_columns(data_type=data_type)

        logger.debug(f"Changed Names: {self.et_data.columns}")
        # despike variables and remove long, flat periods
        self.clean_columns()
        logger.debug(f"Despiked: {self.et_data.head(5)}")
        logger.debug(f"Despiked: {self.et_data.tail(5)}")
        # switch tau sign
        self.tau_fixer()

        # turn decimals in SWC to percent
        self.fix_swc_percent()

        # rescale the quality values to a 0-2 scale
        self.ssitc_scale()

        if "ET_SSITC_TEST" in self.et_data.columns:

            logger.debug(f"SSITC Values: {self.et_data['ET_SSITC_TEST'].unique()}")

            logger.info(f"SSITC Values: {self.et_data['ET_SSITC_TEST'].unique()}")

            logger.info(f"SSITC Values: {self.et_data['ET_SSITC_TEST'].unique()}")

        self.drop_extras()

        logger.debug(f"Extras: {self.et_data.head(5)}")
        logger.debug(f"Extras: {self.et_data.tail(5)}")

        if drop_soil:
            self.et_data = self.remove_extra_soil_params(self.et_data)

        self.et_data = self.et_data.fillna(value=int(-9999))
        logger.debug(f"Fillna: {self.et_data.head(5)}")
        logger.debug(f"Fillna: {self.et_data.tail(5)}")

        self.et_data = self.et_data.replace(-9999.0, int(-9999))
        logger.debug(f"Replace: {self.et_data.head(5)}")
        logger.debug(f"Replace: {self.et_data.tail(5)}")

        if "ET" in self.et_data.columns:
            count_neg_9999 = (self.et_data["ET"] == -9999).sum()

            logger.debug(f"Null Value Count in ET: {count_neg_9999}")
            logger.debug(f"Length of ET: {len(self.et_data['ET'])}")

            logger.info(f"Null Value Count in ET: {count_neg_9999}")
            logger.info(f"Length of ET: {len(self.et_data['ET'])}")

        self.col_order()

    def remove_extra_soil_params(self, df):
        """
        Removes extra soil parameters from the given dataframe.

        :param df: A pandas dataframe containing soil parameters.
        :type df: pandas.DataFrame
        :return: The input dataframe with extra soil parameters removed.
        :rtype: pandas.DataFrame
        """
        # get a list of columns and split them into parts (parameter number number number)
        # am = AmerifluxDataProcessor()
        for col in df.columns:
            collist = col.split("_")
            main_var = collist[0]

            # get rid of columns that don't follow the typical pattern
            if len(collist) > 3 and collist[2] not in ["N", "S"]:
                depth_var = int(collist[2])
                if main_var in ["SWC", "TS", "EC", "K"] and (
                    depth_var >= 1 and int(collist[1]) >= 3
                ):
                    df = df.drop(col, axis=1)
            # drop cols from a specified list math_soils_v2
            elif col in self.MATH_SOILS_V2[:-4]:
                df = df.drop(col, axis=1)
            elif main_var in ["VWC", "Ka"] or "cm_N" in col or "cm_S" in col:
                df = df.drop(col, axis=1)
        return df

    def drop_extras(self):
        """Drops columns specified in the configuration."""
        for col in self.DROP_COLS:
            if col in self.et_data.columns:
                self.et_data = self.et_data.drop(col, axis=1)

    def col_order(self):
        """Puts priority columns first"""
        first_cols = ["TIMESTAMP_END", "TIMESTAMP_START"]
        for col in first_cols:
            ncol = self.et_data.pop(col)
            self.et_data.insert(0, col, ncol)
        logger.debug(f"Column Order: {self.et_data.columns}")
        logger.debug(f"Column Order: {self.et_data.head(5)}")
        logger.debug(f"Column Order: {self.et_data.tail(5)}")

    def datefixer(self, et_data):
        """
        Fixes the date and time format in the given data.

        :param et_data: A pandas DataFrame containing the data to be fixed.
        :return: The fixed pandas DataFrame.

        The `datefixer` method takes a pandas DataFrame `et_data` as input and performs the following operations to fix the date and time format:
        1. Converts the 'TIMESTAMP_START' and 'TIMESTAMP_END' columns to datetime format using the format "%Y%m%d%H%M" and assigns them to 'datetime_start' and 'datetime_end' columns in `et_data`.
        2. Removes duplicate rows in `et_data` based on the 'TIMESTAMP_START' and 'TIMESTAMP_END' columns.
        3. Sets the 'datetime_start' column as the index of `et_data` and sorts the DataFrame based on this index.
        4. Removes any duplicate rows in `et_data` based on the 'datetime_start' column, keeping only the first occurrence.
        5. Resamples the DataFrame at 30-minute intervals and interpolates missing values using linear method.
        6. Returns the fixed pandas DataFrame.

        Example usage:
        ```python
        import pandas as pd

        # Create the input DataFrame et_data
        et_data = pd.DataFrame({
            'TIMESTAMP_START': ['202201011200', '202201011300', '202201020900'],
            'TIMESTAMP_END': ['202201011400', '202201011500', '202201021000']
        })

        # Create an instance of the class containing the datefixer method
        date_fixer = DateFixer()

        # Call the datefixer method to fix the date and time format in et_data
        fixed_data = date_fixer.datefixer(et_data)

        print(fixed_data)
        ```
        """
        # drop null index values

        # create datetime fields to conduct datetime operations on dataset
        et_data["datetime_start"] = pd.to_datetime(
            et_data["TIMESTAMP_START"],
            format="%Y%m%d%H%M",
            errors="coerce",
        )
        et_data["datetime_start"] = et_data["datetime_start"].dropna()
        et_data = et_data.drop_duplicates(subset=["TIMESTAMP_START", "TIMESTAMP_END"])
        et_data = et_data.set_index(["datetime_start"]).sort_index()

        # drop rows with duplicate datetimes
        et_data = et_data[~et_data.index.duplicated(keep="first")]

        # eliminate implausible dates that are set in the future
        et_data = et_data[
            et_data.index <= datetime.datetime.today() + pd.Timedelta(days=1)
        ]

        # eliminate object columns from dataframe before fixing time offsets
        if "TIMESTAMP" in et_data.columns:
            et_data = et_data.drop("TIMESTAMP", axis=1)

        et_data = et_data.infer_objects(copy=False)

        # fix time offsets to harmonize sample frequency to 30 min
        et_data = (
            et_data.resample("30min")  # Directly resample to 30-minute intervals
            .asfreq()  # Establishes a 30-minute frequency grid with NaNs in missing slots
            .interpolate(method="linear", limit=1)  # Fill up to 30-minute gaps
        )

        et_data = et_data.drop(columns=["datetime_start"], errors="ignore")
        et_data = et_data.drop(columns=["datetime_start"], errors="ignore")
        et_data = et_data.reset_index()
        et_data = et_data.dropna(subset=["datetime_start"])
        et_data = et_data.set_index("datetime_start")
        # et_data = et_data[et_data.index.notnull()]

        # remake timestamp marks to match datetime index, filling in NA spots
        et_data["TIMESTAMP_START"] = et_data.index.strftime("%Y%m%d%H%M").astype(
            np.int64
        )

        et_data["TIMESTAMP_END"] = (
            (et_data.index + pd.Timedelta("30min"))
            .strftime("%Y%m%d%H%M")
            .astype(np.int64)
        )

        return et_data

    def update_dataframe_column(self, new_column, old_column):
        """
        Combines new and old columns, retaining maximum values, and removes the old column.

        Args:
            new_column (str): Name of the new column.
            old_column (str): Name of the old column to replace.
        """
        self.et_data[new_column] = self.et_data[[old_column, new_column]].max(axis=1)
        self.et_data = self.et_data.drop(old_column, axis=1)

    def rename_columns(self, data_type="eddy"):
        """Renames columns based on the dataset type and configuration mappings."""

        if data_type == "eddy":
            mappings = self.COL_NAME_MATCH
        else:
            mappings = self.MET_RENAMES

        for old_col, new_col in mappings.items():
            if old_col in self.et_data.columns:
                if new_col in self.et_data.columns:
                    logger.debug(f"Updating column: {old_col} to {new_col}")
                    self.et_data[new_col] = self.et_data[[old_col, new_col]].max(axis=1)
                    self.et_data = self.et_data.drop(old_col, axis=1)
                else:
                    logger.debug(f"Renaming column: {old_col} to {new_col}")
                    self.et_data = self.et_data.rename(columns={old_col: new_col})

    def scale_and_convert(self, column: pd.Series) -> pd.Series:
        """
        Scale the values and then convert them to float type
        :param column: Pandas series or dataframe column
        :return: Scaled and converted dataframe column
        """
        # match rating to new rating
        column = column.apply(self.rating)
        # output at integer
        return column

    def ssitc_scale(self):
        """
        Scale the values in the SSITC columns of et_data dataframe.
        :return: None
        """
        ssitc_columns = [
            "FC_SSITC_TEST",
            "LE_SSITC_TEST",
            "ET_SSITC_TEST",
            "H_SSITC_TEST",
            "TAU_SSITC_TEST",
        ]
        for column in ssitc_columns:
            if column in self.et_data.columns:
                if self.et_data[column].max() > 3:
                    self.et_data[column] = self.scale_and_convert(self.et_data[column])

    @staticmethod
    def rating(x):
        """
        Convert a value into a rating category.

        :param x: The value to be converted.
        :type x: int or float
        :return: The rating category based on the value.
        :rtype: int

        The `rating` method takes a numeric value, `x`, as input and categorizes it into one of
        three rating categories. For values less than or equal to 3, the method returns 0. For values
        between 4 and 6 (inclusive), the method returns 1. For all other values, the method returns 2.
        """
        if 0 <= x <= 3:
            x = 0
        elif 4 <= x <= 6:
            x = 1
        else:
            x = 2
        return x

    def _extract_variable_name(self, variable):
        """
        Extracts the variable name based on given variable.

        :param variable: The variable to extract the variable name from.
        :type variable: str
        :return: The extracted variable name.
        :rtype: str
        """
        if variable in self.OTHER_VARS:
            varlimvar = variable
        elif any(variable in x for x in self.OTHER_VARS):
            temp = variable.split("_")[:2]
            varlimvar = "_".join(temp)
        else:
            varlimvar = variable.split("_")[0]

        return varlimvar

    def replace_out_of_range_with_nan(self, df, variable, replace_w=np.nan):
        """
        Replace values in a specified column with np.nan if they exceed a given range.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name where the replacement should be applied.
        lower_bound (float): The lower bound of the range.
        upper_bound (float): The upper bound of the range.

        Returns:
        pd.DataFrame: The DataFrame with values replaced by np.nan.
        """

        varlimvar = self._extract_variable_name(variable)
        if varlimvar in self.varlimits.index:
            if variable in df.columns:
                upper_bound = self.varlimits.loc[varlimvar, "Max"]
                lower_bound = self.varlimits.loc[varlimvar, "Min"]
                valid_mask = (df[variable] >= lower_bound) & (
                    df[variable] <= upper_bound
                )
                df.loc[~valid_mask, variable] = replace_w
                # df[variable] = df[variable].apply(
                #    lambda x: replace_w if x < lower_bound or x > upper_bound else x
                # )
        return df

    def extreme_limiter(self, df, replace_w=np.nan):
        """
        :param df: pandas DataFrame object containing the data to be processed
        :param replace_w: value to replace the extreme values with (default is np.nan)
        :return: None

        This method takes a pandas DataFrame object (df) and a value (replace_w) as parameters. It iterates through each column of the DataFrame and converts the values to numeric type using `pd.to_numeric` function, with errors set to 'coerce' to replace invalid values with NaN.

        The method then extracts the variable name from the column name using the `_extract_variable_name` method. Finally, it calls the `_check_and_replace` method to check for extreme values and replace them with the given value (replace_w).
        """
        for variable in df.columns:
            df = self.replace_out_of_range_with_nan(df, variable, replace_w)
        return df

    def despike(self, arr):
        """Removes spikes from an array of values based on a specified deviation from the mean.

        Args:
            arr: array of values with spikes
            nstd: number of standard deviations from mean; default is 4.5

        Returns:
            Array of despiked values
        Notes:
            * Not windowed.
            * This method is fast but might be too agressive if the spikes are small relative to seasonal variability.
        """

        stdd = np.nanstd(arr) * self.spike_threshold
        avg = np.nanmean(arr)
        avgdiff = stdd - np.abs(arr - avg)
        y = np.where(avgdiff >= 0, arr, np.nan)

        return y

    def tau_fixer(self):
        """
        Fixes the values in the 'TAU' column of the et_data dataframe by multiplying them by -1.

        :return: None
        """
        if "TAU" in self.et_data.columns:
            self.et_data["TAU"] = -self.et_data["TAU"]

    def fix_swc_percent(self):
        """
        Applies a fix to the SWC columns in the et_data DataFrame.
        This method identifies columns containing "SWC" in their names,
        and checks whether the maximum value is less than 1.5. If it is, the
        values in the column are multiplied by 100, except for any values
        that are -9999.
        :return: None
        """
        for col in self.et_data.columns:
            if "SWC" in col and self.et_data[col].max() < 1.5:
                self.et_data.loc[self.et_data[col] > -9999, col] *= 100

    def timestamp_reset(self):
        """Resets timestamp columns according to DataFrame index."""
        self.et_data["TIMESTAMP_START"] = self.et_data.index
        self.et_data["TIMESTAMP_END"] = self.et_data.index + pd.Timedelta(minutes=30)

    def despike_ewma_fb(self, df_column, span, delta):
        """Apply forwards, backwards exponential weighted moving average (EWMA) to df_column.
        Remove data from df_spikey that is > delta from fbewma.

        Args:
            df_column: pandas Series of data with spikes
            span: size of window of spikes
            delta: threshold of spike that is allowable

        Returns:
            despiked data

        Notes:
            https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python
        """
        # Forward EWMA.
        fwd = pd.Series.ewm(df_column, span=span).mean()
        # Backward EWMA.
        bwd = pd.Series.ewm(df_column[::-1], span=span).mean()
        # Add and take the mean of the forwards and backwards EWMA.
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        np_fbewma = np.mean(stacked_ewma, axis=0)
        np_spikey = np.array(df_column)
        # np_fbewma = np.array(fb_ewma)
        cond_delta = np.abs(np_spikey - np_fbewma) > delta
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        return np_remove_outliers


def load_data():
    df = pd.read_csv("./data/extreme_values.csv")
    return df


def outfile(df, stationname, out_dir):
    """Outputs file following the Ameriflux file naming format
    Args:
        df: the DataFrame that contains the data to be written to a file
        stationname: the name of the station for which the data is being written
        out_dir: the output directory where the file will be saved
    """
    first_index = pd.to_datetime(df.iloc[0, 0], format="%Y%m%d%H%M")
    last_index = pd.to_datetime(
        df.iloc[-1, 1], format="%Y%m%d%H%M"
    )  # df.index[-1], format ='%Y%m%d%H%M')
    filename = (
        stationname
        + f"_HH_{first_index.strftime('%Y%m%d%H%M')}_{last_index.strftime('%Y%m%d%H%M')}.csv"
    )
    df.to_csv(out_dir + stationname + "/" + filename, index=False)


if __name__ == "__main__":
    data = load_data()
