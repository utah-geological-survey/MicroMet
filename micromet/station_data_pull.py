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
