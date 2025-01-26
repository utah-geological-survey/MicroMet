import requests
import datetime
from requests.auth import HTTPBasicAuth
import pandas as pd
from io import BytesIO
from .converter import Reformatter


def get_times(station, config, loggertype="eddy"):
    ip = config[station]["ip"]
    if loggertype == "eddy":
        port = config[station]["eddy_port"]
    else:
        port = config[station]["met_port"]

    clk_url = f"http://{ip}:{port}/?"
    # url = f"http://{ip}/tables.html?command=DataQuery&mode=since-record&format=toA5&uri=dl:Flux_AmeriFluxFormat&p1=0"
    clk_args = {
        "command": "ClockCheck",
        "uri": "dl",
        "format": "json",
    }
    clktimeresp = requests.get(
        clk_url,
        params=clk_args,
        auth=HTTPBasicAuth(config["LOGGER"]["login"], config["LOGGER"]["pw"]),
    ).json()
    if "time" in clktimeresp.keys():
        clktime = clktimeresp["time"]
    else:
        clktime = None

    comptime = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    return clktime, comptime


def get_station_data(station, config, reformat=True, loggertype="eddy"):
    ip = config[station]["ip"]

    if loggertype == "eddy":
        tabletype = "Flux_AmeriFluxFormat"
        port = config[station]["eddy_port"]
    else:
        tabletype = "Statistics_AmeriFlux"
        port = config[station]["met_port"]

    url = f"http://{ip}:{port}/tables.html?"
    params = {
        "command": "DataQuery",
        "mode": "since-record",
        "format": "toA5",
        "uri": f"dl:{tabletype}",
        "p1": "0",
    }

    response_1 = requests.get(
        url,
        params=params,
        auth=HTTPBasicAuth(config["LOGGER"]["login"], config["LOGGER"]["pw"]),
    )

    if response_1.status_code == 200:
        headers = pd.read_csv(BytesIO(response_1.content), skiprows=[0]).iloc[0:2, :].T
        raw_data = pd.read_csv(BytesIO(response_1.content), skiprows=[0, 2, 3])
        pack_size = len(response_1.content) * 1e-6

        if raw_data is not None:
            if reformat is True:
                am_data = Reformatter(raw_data)
                am_df = am_data.et_data
            else:
                am_df = raw_data

            return am_df, pack_size
    else:
        print(f"Error: {response_1.status_code}")
        return None, None


def remove_existing_records(df1, column_to_check, values_to_remove):
    """
    Remove rows from df1 where the specified column values exist in the given list.

    Parameters:
    df1 (pd.DataFrame): The DataFrame to be filtered.
    column_to_check (str): The column name to check for values to remove.
    values_to_remove (list): List of values to be removed from the DataFrame.

    Returns:
    pd.DataFrame: A filtered DataFrame excluding matching rows.
    """
    # Ensure the specified column exists in the DataFrame
    if column_to_check in df1.columns:
        print(f"Column '{column_to_check}' found in DataFrame")
        newcol = column_to_check
    if column_to_check.upper() in df1.columns:
        print(f"Column '{column_to_check}' found in DataFrame, trying lowercase")
        newcol = column_to_check.upper()
    elif column_to_check.lower() in df1.columns:
        print(f"Column '{column_to_check}' found in DataFrame, trying lowercase")
        newcol = column_to_check.lower()
    else:
        raise ValueError(f"Column '{column_to_check}' not found in DataFrame")

    # Filter out rows where the specified column has values in the list
    df_filtered = df1[~df1[newcol].isin(values_to_remove)]

    return df_filtered


def compare_sql_to_station(
    df, station, engine, field="timestamp_end", loggertype="eddy"
):
    # SQL query to get the max value from a field
    if loggertype == "eddy":
        table = "amfluxeddy"
    else:
        table = "amfluxmet"

    query = f"SELECT {field} FROM {table} WHERE stationid = '{station}';"

    # Execute query and fetch results into a DataFrame
    exist = pd.read_sql(query, con=engine)

    existing = exist["timestamp_end"].values

    df_filtered = remove_existing_records(df, field, existing)

    return df_filtered


def get_max_date(station, engine, loggertype="eddy"):
    # SQL query to get the max value from a field
    if loggertype == "eddy":
        table = "amfluxeddy"
    else:
        table = "amfluxmet"

    query = f"SELECT MAX(timestamp_end) AS max_value FROM {table} WHERE stationid = '{station}';"

    # Execute query and fetch results into a DataFrame
    df = pd.read_sql(query, con=engine)

    # Access the max value
    max_value = df["max_value"].iloc[0]

    return max_value


def stat_dl_con_ul(site_folders, config, engine):
    for stationid, name in site_folders.items():
        station = stationid.split("-")[-1]
        for dat in ["eddy", "met"]:
            if dat in config[station].keys():
                stationtime, comptime = get_times(station, config, loggertype=dat)
                am_df, pack_size = get_station_data(station, config, loggertype=dat)

                if am_df is not None:
                    am_df_filt = compare_sql_to_station(
                        am_df, station, engine, loggertype=dat
                    )
                    mindate = am_df_filt["TIMESTAMP_START"].min()
                    maxdate = am_df_filt["TIMESTAMP_START"].min()
                    raw_len = len(am_df)
                    am_df_len = len(am_df_filt)
                    am_df_filt = am_df_filt.rename(columns=str.lower)
                    am_df_filt.to_sql(
                        f"amflux{dat}", con=engine, if_exists="append", index=False
                    )
                    # Define variable names and values
                    variables = {
                        "stationid": stationid,
                        "talbetype": dat,
                        "mindate": mindate,
                        "maxdate": maxdate,
                        "datasize_mb": pack_size,
                        "stationdf_len": raw_len,
                        "uploaddf_len": am_df_len,
                        "stationtime": stationtime,
                        "comptime": comptime,
                    }

                    # Create a single-row dataframe
                    df = pd.DataFrame([variables])
                    df.to_sql(
                        f"uploadstats", con=engine, if_exists="append", index=False
                    )
                    # Display the dataframe
                else:
                    mindate = None
                    maxdate = None
                    raw_len = None
                    am_df_len = None

                print(dat)
                print(f"Station {station}")
                print(f"Mindate {mindate}  Maxdate {maxdate}")
                print(f"data size = {pack_size}")
                print(f"{am_df_len} vs {raw_len} rows")
