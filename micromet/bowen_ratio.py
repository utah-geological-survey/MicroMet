import numpy as np
import pandas as pd

import ec

# Define constants
k = 0.4  # von Kármán constant

class BowenRatioRecord(object):

    def __init__(self, **kwargs):
        self.eps = 0.622

        self.t_high = 0 #Air Temperature at an Upper Height (C)
        self.t_low = 0 #ir Temperature at a Lower Height (C)
        self.e_high = 0 # Air Vapor Pressure at an Upper Height (kPa)
        self.e_low = 0 # Air Vapor Pressure at a Lower Height (kPa)
        self.P = 0 # Air Pressure (kPa)
        self.Rn = 0 # Net Radiation (W/m2)
        self.G = 0 # Soil Heat Flux (W/m2)
        self.m_soil = 0 # Volumetric Water Content (unitless)
        self.t_soil = 0 # Soil Temperature (C)
        self.br = 0 #Bowen Ratio (unitless)
        self.H = 0 #Sensible Heat Flux (W/m2)
        self.lambdaE = 0 #Latent Heat Flux (W/m2)
        self.ET = 0 #Evapotranspiration (in)

    def get_dates(self, df):
        """Gets the start date, end date, and date length from a DataFrame with a datetime index.

        Args:
            df (DataFrame): The DataFrame with a datetime index.

        """
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            self.startdate = df.first_valid_index()
            self.enddate = df.last_valid_index()
            self.datelength = self.enddate - self.startdate
        else:
            print('index must be datetime')

    def clean_records(self,df, freq='1H'):
        """
        Args:
            df: The input dataframe containing records to be cleaned.
            freq: The frequency at which the resampling needs to be done. The default value is '1H'.

        Returns:
            The cleaned dataframe with resampled records based on the specified frequency.

        Example:
            df = clean_records(df, freq='2H')
        """
        return df.resample('15min').asfreq().interpolate(method='time').resample(freq)

    def calculate_evapotranspiration(self, df):
        """
        Calculate evapotranspiration based on the given DataFrame.

        Args:
            df (DataFrame): Input DataFrame containing the following columns:
                - t_high (float): High temperature (in Celsius)
                - t_low (float): Low temperature (in Celsius)
                - e_high (float): High vapor pressure (in hPa)
                - e_low (float): Low vapor pressure (in hPa)
                - P (float): Atmospheric pressure (in hPa)
                - Cp (float): Heat capacity of air at constant pressure (in J/kg*K)
                - PA (float): Air density (in kg/m^3)
                - Rn (float): Net radiation (in W/m^2)
                - G (float): Ground heat flux (in W/m^2)
                - eps (float): Atmospheric emissivity

        Returns:
            DataFrame: The input DataFrame with the following additional columns:
                - t_mean (float): Mean temperature (average of t_high and t_low)
                - e_mean (float): Mean vapor pressure (average of e_high and e_low)
                - t_range (float): Temperature range (t_high - t_low)
                - e_range (float): Vapor pressure range (e_high - e_low)
                - cp (float): Heat capacity of air at constant pressure adjusted for vapor pressure
                - lamb (float): Latent heat of vaporization (in J/kg)
                - BowenRatio (float): Bowen ratio
                - ef (float): Evaporative fraction (when BowenRatio is between -0.75 and -1.25)
                - lambdaE (float): Latent heat flux (when BowenRatio is between -0.75 and -1.25)
                - ET (float): Evapotranspiration (when BowenRatio is between -0.75 and -1.25)

        Note:
            This method assumes that the input DataFrame contains all the necessary columns.
        """
        df['t_mean'] = df[['t_high', 't_low']].mean(axis=1)
        df['e_mean'] = df[['e_high','e_low']].mean(axis=1)
        df['t_range'] = df['t_high'] - df['t_low']
        df['e_range'] = df['e_high'] - df['e_low']
        df['cp'] = 1004.2 + 1845.6 * self.eps * df['e_mean'] / (df['P'] - df['e_mean'])
        df['lamb'] = ec.calc_latent_heat_of_vaporization(df['t_mean'], units='C')
        df['BowenRatio'] = df['Cp'] * df['PA'] * df['t_range'] / (df['lamb'] * self.eps * df['e_range'])
        selection = ((df['BowenRatio']>-0.75) or (df['BowenRatio']<-1.25))
        df['ef'] = np.nan
        df['lambdaE'] = np.nan
        df['ET'] = np.nan
        df.loc[selection, 'ef'] = 1/(1 + df['BowenRatio'])
        df.loc[selection, 'lambdaE'] = df['ef'] * (df['Rn'] + df['G'])
        df.loc[selection, 'ET'] = df['ef'] * (df['Rn'] + df['G'])

    def calc_latent_heat_of_vaporization(self, temperature):
        return 2500800 - 2360 * temperature + 1.6 * temperature**2 - 0.06 * temperature**3

    def calc_water_density(self, temperature):
        return 999.97495 * (1 - (temperature - 3.983035)**2 *(temperature + 301.797) / (522528.9*(temperature + 69.34881)))

    def get_watts_to_water(self, temperature, dayfraction=1/48):
        return (self.calc_water_density() * 86.4 * dayfraction) / (ec.calc_latent_heat_of_vaporization(temperature) * 25.4)



def stability_function_heat(z, L):
    """
    Stability function for heat.
    For simplicity, this example uses a basic approximation.
    """
    if L > 0:
        # Stable conditions
        phi_h = 1 + 5 * (z / L)
    else:
        # Unstable conditions
        phi_h = (1 - 16 * (z / L)) ** -0.5
    return phi_h

def estimate_transfer_coefficient(u_star, L, z):
    """
    Estimate the transfer coefficient K_h using Monin-Obukhov Similarity Theory.
    """
    phi_h = stability_function_heat(z, L)
    K_h = (k * u_star * z) / phi_h
    return K_h

# Example DataFrame
data = {
    'u_star': [0.1, 0.2, 0.15],  # Friction velocity in m/s
    'L': [-100, 200, -50],       # Obukhov length in meters
}

df = pd.DataFrame(data)

# Height
z = 2.0  # Height in meters

# Estimate transfer coefficient
df['K_h'] = df.apply(lambda row: estimate_transfer_coefficient(row, row['u_star'], row['L'], z), axis=1)

print(df)
