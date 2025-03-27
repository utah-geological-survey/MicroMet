# https://www.scielo.br/j/rbcs/a/dFCLs7jXncc98VWNjdT64Xd/
# https://doi.org/10.1590/S0100-06832013000100011
# 10.52547/maco.2.1.5


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Constants
WATER_HEAT_CAPACITY = 4.18  # MJ m-3 K-1


def temperature_gradient(T_upper, T_lower, depth_upper, depth_lower):
    """
    Calculate the temperature gradient (°C/m) between two depths.

    Parameters:
    - T_upper: Temperature at upper depth (°C)
    - T_lower: Temperature at lower depth (°C)
    - depth_upper: Upper sensor depth (m)
    - depth_lower: Lower sensor depth (m)

    Returns:
    - Temperature gradient (°C/m)
    """
    return (T_lower - T_upper) / (depth_lower - depth_upper)


def soil_heat_flux(T_upper, T_lower, depth_upper, depth_lower, k):
    """
    Calculate soil heat flux (G) using temperature gradient and thermal conductivity.

    Parameters:
    - T_upper: Temperature at upper depth (°C)
    - T_lower: Temperature at lower depth (°C)
    - depth_upper: Upper sensor depth (m)
    - depth_lower: Lower sensor depth (m)
    - k: Thermal conductivity (W/(m·°C))

    Returns:
    - Soil heat flux (W/m^2)
    """
    return -k * temperature_gradient(T_upper, T_lower, depth_upper, depth_lower)


def volumetric_heat_capacity(dry_density, moisture_content, organic_matter_content=0):
    """
    Calculate the volumetric heat capacity of soil.

    Parameters:
    - dry_density: Dry density of the soil (Mg/m^3)
    - moisture_content: Volumetric moisture content (m^3/m^3)
    - organic_matter_content: Fraction of organic matter in soil

    Returns:
    - Volumetric heat capacity (MJ m-3 K-1)
    """
    # Heat capacity of dry soil is approximated, can be refined based on soil type
    heat_capacity_dry_soil = 0.84  # MJ m-3 K-1 (approximate)
    heat_capacity_organic_matter = 2.5  #  MJ m-3 K-1
    return (
        (dry_density * heat_capacity_dry_soil)
        + (moisture_content * WATER_HEAT_CAPACITY)
        + (organic_matter_content * heat_capacity_organic_matter)
    )


def diurnal_amplitude(series: pd.Series) -> pd.Series:
    """
    Calculate the amplitude of diurnal fluctuations (daily max - daily min) from a pandas series.

    Parameters:
    series (pd.Series): Pandas Series with a datetime index.

    Returns:
    pd.Series: Daily amplitude values.
    """
    daily_max = series.resample("D").max()
    daily_min = series.resample("D").min()
    amplitude = daily_max - daily_min

    return amplitude


def diurnal_peak_lag(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Calculate the daily lag (offset in time) between peaks of two diurnal pandas series.

    Parameters:
    series1 (pd.Series): First pandas Series with datetime index.
    series2 (pd.Series): Second pandas Series with datetime index.

    Returns:
    pd.Series: Daily peak lag values in hours.
    """

    def daily_peak_time(series):
        return series.resample("D").apply(
            lambda x: x.idxmax().hour + x.idxmax().minute / 60
        )

    peak_time_1 = daily_peak_time(series1)
    peak_time_2 = daily_peak_time(series2)

    peak_lag = peak_time_1 - peak_time_2

    # Adjust lag to account for day wrap-around (e.g., -23 hours to +1 hour)
    peak_lag = peak_lag.apply(lambda x: (x + 12) % 24 - 12)

    return peak_lag


def fit_sinusoid(t, data):
    """
    Fits a sinusoidal curve to the temperature data.
    """
    # Initial guess for the parameters [A, omega, phase, offset]
    guess_amp = np.std(data)
    guess_freq = 2 * np.pi / 86400  # Assuming daily cycle
    guess_phase = 0
    guess_offset = np.mean(data)
    p0 = [guess_amp, guess_freq, guess_phase, guess_offset]

    # Fit the sine curve
    popt, pcov = curve_fit(sinusoid, t, data, p0=p0)
    return popt


def sinusoid(t, A, omega, phase, offset):
    """
    Sinusoidal function.
    """
    return A * np.sin(omega * t + phase) + offset


def calculate_thermal_diffusivity_for_pair(df, col1, col2, z1, z2, period):
    """
    Calculates thermal diffusivity using the log-amplitude, amplitude, and phase methods.

    Parameters:
        df (pd.DataFrame): DataFrame with temperature columns.
        col1 (str): Name of the first depth column.
        col2 (str): Name of the second depth column.
        z1 (float): Depth of the first sensor (m).
        z2 (float): Depth of the second sensor (m).
        period (int): Period of the temperature oscillation (seconds).

    Returns:
        dict: A dictionary containing thermal diffusivity calculated by each method.
    """
    temp_data_depth1 = df[col1].values
    temp_data_depth2 = df[col2].values

    # Time vector in seconds
    time_sec = np.arange(
        0, len(temp_data_depth1) * 1800, 1800
    )  # Assuming 30-min intervals

    # Fit sine curves
    popt_depth1 = fit_sinusoid(time_sec, temp_data_depth1)
    popt_depth2 = fit_sinusoid(time_sec, temp_data_depth2)

    # Extract fitted parameters
    A1, omega1, phase1, offset1 = popt_depth1
    A2, omega2, phase2, offset2 = popt_depth2

    # Log-amplitude method
    alpha_log_amplitude = (
        ((z2 - z1) / np.log(A1 / A2)) ** 2 / (2 * omega1)
        if A1 != 0 and A2 != 0 and np.log(A1 / A2) != 0
        else None
    )

    # Amplitude method
    alpha_amplitude = (
        ((z2 - z1) / (np.log(A1 / A2))) ** 2 / (2 * np.pi / period)
        if A1 != 0 and A2 != 0 and np.log(A1 / A2) != 0
        else None
    )

    # Phase method
    alpha_phase = (
        ((z2 - z1) / (phase2 - phase1)) ** 2 / (2 * np.pi / period)
        if (phase2 - phase1) != 0
        else None
    )

    return {
        "log_amplitude": alpha_log_amplitude,
        "amplitude": alpha_amplitude,
        "phase": alpha_phase,
    }


def calculate_thermal_properties_for_all_pairs(
    df, depth_mapping, period=86400, dry_density=1.3, moisture_content=0.2
):
    """
    Calculates thermal diffusivity, volumetric heat capacity, and soil heat flux for all depth pairs.

    Parameters:
        df (pd.DataFrame): Time-indexed DataFrame with temperature columns for different depths.
        depth_mapping (dict): A dictionary where keys are column names and values are their corresponding depths (in meters).
        period (int): The period of the temperature oscillation (in seconds, default is 24 hours).
        dry_density (float): Dry density of soil (Mg/m^3).
        moisture_content (float): Volumetric moisture content of soil (m^3/m^3).

    Returns:
        pd.DataFrame: A DataFrame containing the calculated thermal properties for each depth pair.
    """
    results = []
    depth_cols = list(depth_mapping.keys())

    # Calculate volumetric heat capacity (assumed constant for all pairs here)
    ch = volumetric_heat_capacity(dry_density, moisture_content)

    for i in range(len(depth_cols)):
        for j in range(i + 1, len(depth_cols)):
            col1 = depth_cols[i]
            col2 = depth_cols[j]
            z1 = depth_mapping[col1]
            z2 = depth_mapping[col2]

            # Calculate thermal diffusivity
            alpha_results = calculate_thermal_diffusivity_for_pair(
                df, col1, col2, z1, z2, period
            )

            # Calculate soil heat flux
            # Using a simple approach: heat flux between the two depths
            k = 0.5  # Approximation, can be improved with site-specific data
            G = soil_heat_flux(
                df[col1].mean(), df[col2].mean(), z1, z2, k
            )  # Using mean temperatures for simplicity

            results.append(
                {
                    "depth1_col": col1,
                    "depth2_col": col2,
                    "z1 (m)": z1,
                    "z2 (m)": z2,
                    "thermal_diffusivity_log_amp (m^2/s)": alpha_results[
                        "log_amplitude"
                    ],
                    "thermal_diffusivity_amplitude (m^2/s)": alpha_results["amplitude"],
                    "thermal_diffusivity_phase (m^2/s)": alpha_results["phase"],
                    "volumetric_heat_capacity (MJ m-3 K-1)": ch,
                    "soil_heat_flux (W/m^2)": G,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("utd_soil_data.csv")
    df["datetime_start"] = pd.to_datetime(df["datetime_start"])
    df.set_index("datetime_start", inplace=True)

    # Define depth mapping
    depth_mapping = {
        "ts_3_1_1": 0.05,  # 5 cm
        "ts_3_2_1": 0.10,  # 10 cm
        "ts_3_3_1": 0.20,  # 20 cm
    }

    # Calculate thermal properties
    results_df = calculate_thermal_properties_for_all_pairs(df, depth_mapping)
    print(results_df)
