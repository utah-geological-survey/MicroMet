# Original scripts in Fortran by Lawrence Hipps USU
# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import pandas as pd
# import scipy
import numpy as np
from scipy import signal
import statsmodels.api as sm
# import dask as dd
# Public Module EC

import numba
from numba import njit, jit
from typing import TypeVar, Generic


# Useful Links to help in solving some calculation issues
# https://stackoverflow.com/questions/47594932/row-wise-interpolation-in-dataframe-using-interp1d
# https://krstn.eu/fast-linear-1D-interpolation-with-numba/
# https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html
# https://pythonawesome.com/maximum-covariance-analysis-in-python/
# https://pyxmca.readthedocs.io/en/latest/quickstart.html#maximum-covariance-analysis
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cov.html
# https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html
# https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acovf.html
# https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.ccovf.html
# https://python-advanced.quantecon.org/index_time_series_models.html
# Style guide for documentation: https://google.github.io/styleguide/pyguide.html

# Allows for specifying data types for input into python functions;
# this custom type allows for single floats or arrays of floats


class CalcFlux(object):
    """Determines H20 flux from input weather data, including a KH20 sensor, by the eddy covariance method.

    Args:
        df: dataframe Weather Parameters for the Eddy Covariance Method; must be time-indexed and include Ux, Uy, Uz, Pr, Ea, and LnKH

    Returns:
        Atmospheric Fluxes

    Notes:
        * No High Pass Filtering or Trend Removal are Applied to the Data
        * Time Series Data Are Moved Forward and Backward to Find Maximum Covariance Values
        * Air Temperature and Sensible Heat Flux are Estimated From Sonic Temperature and Wind Data
        * Other Corrections Include Transducer Shadowing, Traditional Coordinate Rotation, High Frequency Corrections, and WPL"""

    def __init__(self, **kwargs):

        self.Rv = 461.51  # Water Vapor Gas Constant, J/[kg*K]
        self.Ru = 8.3143  # Universal Gas Constant, J/[kg*K]
        self.Cpd = 1005.0  # Specific Heat of Dry Air, J/[kg*K]
        self.Rd = 287.05  # Dry Air Gas Constant, J/[kg*K]
        self.md = 0.02896  # Dry air molar mass, kg/mol
        self.Co = 0.21  # Molar Fraction of Oxygen in the Atmosphere
        self.Mo = 0.032  # Molar Mass of Oxygen (gO2/mole)

        self.Cpw = 1952.0  # specific heat of water vapor at constant pressure [J/(kg K)]
        self.Cw = 4218.0  # specific heat of liquid water at 0 C [J/(kg K)]
        self.epsilon = 18.016 / 28.97  # molecular mass ratio of water vapor to dry air
        self.g = 9.81  # acceleration due to gravity at sea level (m/s^2)
        self.von_karman = 0.41  # von Karman constant (Dyer & Hicker 1970, Webb 1970)
        self.MU_WPL = 28.97 / 18.016  # molecular mass ratio of dry air to water vapor (used in WPL correction)
        self.Omega = 7.292e-5  # Angular velocity of the earth for calculation of Coriolis Force (2PI/sidreal_day, where sidereal day = 23 hr 56 min. [rad/s]
        self.Sigma_SB = 5.6718e-8  # Stefan-Boltzmann constant in air [J/(K^4 m^2 s), see page 336 in McGee (1988)]

        self.meter_type = 'IRGASON'
        self.XKH20 = 1.412  # Path Length of KH20, cm
        self.XKwC1 = -0.152214126  # First Order Coefficient in Vapor Density-KH20 Output Relationship, cm
        self.XKwC2 = -0.001667836  # Second Order Coefficient in Vapor Density-KH20 Output Relationship, cm

        self.sonic_dir = 225  # Degrees clockwise from true North
        self.UHeight = 3.52  # Height of Sonic Anemometer above Ground Surface, m
        self.PathDist_U = 0.0  # Separation Distance Between Sonic Anemometer and KH20 or IRGA unit, m, 0.1

        self.lag = 10  # number of lags to consider
        self.direction_bad_min = 0  # Clockwise Orientation from DirectionKH20_U
        self.direction_bad_max = 360  # Clockwise Orientation from DirectionKH20_U

        self.Kw = 1  # Extinction Coefficient of Water (m^3/[g*cm]) -instrument calibration
        self.Ko = -0.0045  # Extinction Coefficient of Oxygen (m^3/[g*cm]) -derived experimentally

        self.covar = {}  # Dictionary of covariance values; includes max calculated covariances
        self.avgvals = {}  # Dictionary of the average values

        # List of Variables for despiking
        self.despikefields = ['Ux', 'Uy', 'Uz', 'Ts', 'volt_KH20', 'Pr', 'Ta', 'Rh']

        # Allow for update of input parameters
        # https://stackoverflow.com/questions/60418497/how-do-i-use-kwargs-in-python-3-class-init-function
        self.__dict__.update(kwargs)

        # List of common variables and their units
        self.parameters = {
            'Ea': ['Actual Vapor Pressure', 'kPa'],
            'LnKH': ['Natural Log of Krypton Hygrometer Output', 'ln(mV)'],
            'Pr': ['Air Pressure', 'Pa'],
            'Ta': ['Air Temperature', 'K'],
            'Ts': ['Sonic Temperature', 'K'],
            'Ux': ['X Component of Wind Speed', 'm/s'],
            'Uy': ['Y Component of Wind Speed', 'm/s'],
            'Uz': ['Z Component of Wind Speed', 'm/s'],
            'E': ['Vapor Pressure', 'kPa'],
            'Q': ['Specific Humidity', 'unitless'],
            'pV': ['Water Vapor Density', 'kg/m^3'],
            'Sd': ['Entropy of Dry Air', 'J/K'],
            'Tsa': ['Absolute Air Temperature Derived from Sonic Temperature', 'K'],
        }

    def runall(self, df: pd.DataFrame) -> pd.Series:
        """Runs through complete processing of eddy covariance dataset.

        Args:
            df: dataframe Weather Parameters for the Eddy Covariance Method;
            must be time-indexed and include Ux, Uy, Uz, Pr, Ea, and pV or LnKH

        Returns:
            A series of values aggregated over the length of the provided dataframe

        """

        df = self.renamedf(df)

        if 'Ea' in df.columns:
            pass
        else:
            df['Ea'] = self.tetens(df['Ta'].to_numpy())

        if self.meter_type == 'IRGASON':
            pass
        else:
            if 'LnKH' in df.columns:
                pass
            elif 'volt_KH20' in df.columns:
                df['LnKH'] = np.log(df['volt_KH20'].to_numpy())
            # Calculate the Correct XKw Value for KH20
            XKw = self.XKwC1 + 2 * self.XKwC2 * (df['pV'].mean() * 1000.)
            self.Kw = XKw / self.XKH20
            # TODO Calc pV from lnKH20 and add to dataframe as variable

        for col in self.despikefields:
            if col in df.columns:
                df[col] = self.despike(df[col].to_numpy(), nstd=4.5)

        # Convert Sonic and Air Temperatures from Degrees C to Kelvin
        df.loc[:, 'Ts'] = self.convert_CtoK(df['Ts'].to_numpy())
        df.loc[:, 'Ta'] = self.convert_CtoK(df['Ta'].to_numpy())

        # Remove shadow effects of the CSAT (this is also done by the CSAT Firmware)
        df['Ux'], df['Uy'], df['Uz'] = self.shadow_correction(df['Ux'].to_numpy(),
                                                              df['Uy'].to_numpy(),
                                                              df['Uz'].to_numpy())
        self.avgvals = df.mean().to_dict()

        # Calculate Sums and Means of Parameter Arrays
        df = self.calculated_parameters(df)

        # Calculate Covariances (Maximum Furthest From Zero With Sign in Lag Period)
        self.calc_covar(df['Ux'].to_numpy(),
                        df['Uy'].to_numpy(),
                        df['Uz'].to_numpy(),
                        df['Ts'].to_numpy(),
                        df['Q'].to_numpy())

        # Calculate max variance to close separation between sensors
        velocities = {"Ux": df["Ux"].to_numpy(),
                      "Uy": df["Uy"].to_numpy(),
                      "Uz": df["Uz"].to_numpy()}

        covariance_variables = {"Ux": df["Ux"].to_numpy(),
                                "Uy": df["Uy"].to_numpy(),
                                "Uz": df["Uz"].to_numpy(),
                                "Ts": df["Ts"].to_numpy(),
                                "pV": df["pV"].to_numpy(),
                                "Q": df["Q"].to_numpy(),
                                "Sd": df["Sd"].to_numpy()}

        # This iterates through the velocities and calculates the maximum covariance between
        # the velocity and the other variables
        for ik, iv in velocities.items():
            for jk, jv in covariance_variables.items():
                self.covar[f"{ik}-{jk}"] = self.calc_max_covariance(iv, jv)[0][1]

        self.covar["Ts_Q"] = self.calc_max_covariance(df, 'Ts', 'Q', self.lag)[0][1]

        # Traditional Coordinate Rotation
        cosν, sinν, sinTheta, cosTheta, Uxy, Uxyz = self.coord_rotation(df)

        # Find the Mean Squared Error of Velocity Components and Humidity
        self.UxMSE = self.calc_MSE(df['Ux'])
        self.UyMSE = self.calc_MSE(df['Uy'])
        self.UzMSE = self.calc_MSE(df['Uz'])
        self.QMSE = self.calc_MSE(df['Q'])

        # Correct Covariances for Coordinate Rotation
        self.covar_coord_rot_correction(cosν, sinν, sinTheta, cosTheta)

        Ustr = np.sqrt(self.covar["Uxy_Uz"])

        # Find Average Air Temperature From Average Sonic Temperature
        Tsa = self.calc_Tsa(df['Ts'].mean(), df['Pr'].mean(), df['pV'].mean())

        # Calculate the Latent Heat of Vaporization (eq. 2.57 in Foken)
        lamb = (2500800 - 2366.8 * (self.convert_KtoC(Tsa)))

        # Determine Vertical Wind and Water Vapor Density Covariance
        Uz_pV = (self.covar["Uz_pV"] / XKw) / 1000

        # Calculate the Correct Average Values of Some Key Parameters
        self.Cp = self.Cpd * (1 + 0.84 * df['Q'].mean())
        self.pD = (df['Pr'].mean() - df['E'].mean()) / (self.Rd * Tsa)
        self.p = self.pD + df['pV'].mean()

        # Calculate Variance of Air Temperature From Variance of Sonic Temperature
        StDevTa = np.sqrt(self.covar["Ts_Ts"]
                          - 1.02 * df['Ts'].mean() * self.covar["Ts_Q"]
                          - 0.2601 * self.QMSE * df['Ts'].mean() ** 2)
        Uz_Ta = self.covar["Uz_Ts"] - 0.07 * lamb * Uz_pV / (self.p * self.Cp)

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td_dewpoint(df['E'].mean())
        D = self.calc_Es_sat_vapor_pressure(Tsa) - df['E'].mean()
        S = (self.calc_Q_specific_humidity(df['Pr'].mean(),
                                           self.calc_Es(Tsa + 1)) - self.calc_Q(df['Pr'].mean(),
                                                                                self.calc_Es(Tsa - 1))) / 2

        # Determine Wind Direction
        Ux_avg = np.mean(df['Ux'].to_numpy())
        Uy_avg = np.mean(df['Uy'].to_numpy())
        Uz_avg = np.mean(df['Uz'].to_numpy())

        pathlen, direction = self.determine_wind_dir(Ux_avg, Uy_avg)

        # Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df['Uz'].std()
        UMean = Ux_avg * cosTheta * cosν + Uy_avg * cosTheta * sinν + Uz_avg * sinTheta

        # Frequency Response Corrections (Massman, 2000 & 2001)
        tauB = 3600 / 2.8
        tauEKH20 = np.sqrt((0.01 / (4 * UMean)) ** 2 + (pathlen / (1.1 * UMean)) ** 2)
        tauETs = np.sqrt((0.1 / (8.4 * UMean)) ** 2)
        tauEMomentum = np.sqrt((0.1 / (5.7 * UMean)) ** 2 + (0.1 / (2.8 * UMean)) ** 2)

        # Calculate ζ and Correct Values of Uᕽ and Uz_Ta
        L = self.calc_L(Ustr, Tsa, Uz_Ta)
        alpha, X = self.calc_AlphX(L)
        fX = X * UMean / self.UHeight
        B = 2 * np.pi * fX * tauB
        momentum = 2 * np.pi * fX * tauEMomentum
        _Ts = 2 * np.pi * fX * tauETs
        _KH20 = 2 * np.pi * fX * tauEKH20
        Ts = self.correct_spectral(B, alpha, _Ts)
        self.covar["Uxy_Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy_Uz"])

        # Recalculate L With New Uᕽ and Uz_Ta, and Calculate High Frequency Corrections
        L = self.calc_L(Ustr, Tsa, Uz_Ta / Ts)
        alpha, X = self.calc_AlphX(L)
        Ts = self.correct_spectral(B, alpha, _Ts)
        KH20 = self.correct_spectral(B, alpha, _KH20)

        # Correct the Covariance Values
        Uz_Ta /= Ts
        Uz_pV /= KH20
        self.covar["Uxy_Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy_Uz"])
        self.covar["Uz_Sd"] /= KH20
        exchange = ((self.p * self.Cp) / (S + self.Cp / lamb)) * self.covar["Uz_Sd"]

        # KH20 Oxygen Correction
        Uz_pV += self.correct_KH20(Uz_Ta, df['Pr'].mean(), Tsa)

        # Calculate New H and LE Values
        H = self.p * self.Cp * Uz_Ta
        lambdaE = lamb * Uz_pV

        # Webb, Pearman and Leuning Correction
        pVavg = np.mean(df['pV'].to_numpy())
        lambdaE = self.webb_pearman_leuning(lamb, Tsa, pVavg, Uz_Ta, Uz_pV)

        # Finish Output
        Tsa = self.convert_KtoC(Tsa)
        Td = self.convert_KtoC(Td)
        zeta = self.UHeight / L
        ET = lambdaE * self.get_Watts_to_H2O_conversion_factor(Tsa, (
                df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(1, unit='D'))
        # Out.Parameters = CWP
        self.columns = ['Ta', 'Td', 'D', 'Ustr', 'zeta', 'H', 'StDevUz', 'StDevTa', 'direction', 'exchange', 'lambdaE',
                        'ET', 'Uxy']
        self.out = [Tsa, Td, D, Ustr, zeta, H, StDevUz, StDevTa, direction, exchange, lambdaE, ET, Uxy]
        return pd.Series(data=self.out, index=self.columns)

    def determine_wind_dir(self, uxavg, uyavg):
        # Determine Wind Direction

        v = np.sqrt(uxavg ** 2 + uyavg ** 2)
        wind_dir = np.arctan(uyavg / uxavg) * 180.0 / np.pi
        if uxavg < 0:
            if uyavg >= 0:
                wind_dir += wind_dir + 180.0
            else:
                wind_dir -= wind_dir - 180.0
        wind_compass = -1.0 * wind_dir + self.sonic_dir
        if wind_compass < 0:
            wind_compass += 360
        elif wind_compass > 360:
            wind_compass -= 360

        # Calculate the Lateral Separation Distance Projected Into the Mean Wind Direction
        pathlen = self.PathDist_U * np.abs(np.sin((np.pi / 180) * wind_compass))
        return pathlen, wind_compass

    def covar_coord_rot_correction(self, cosν, sinν, sinTheta, cosTheta):
        """Correct Covariances for Coordinate Rotation

        Args:
            cosν:
            sinν:
            sinTheta:
            cosTheta:

        Returns:

        """
        #
        Uz_Ts = self.covar["Uz_Ts"] * cosTheta - self.covar["Ux_Ts"] * sinTheta * cosν \
                - self.covar["Uy_Ts"] * sinTheta * sinν
        if np.abs(Uz_Ts) >= np.abs(self.covar["Uz_Ts"]):
            self.covar["Uz_Ts"] = Uz_Ts

        Uz_pV = self.covar["Uz_pV"] * cosTheta - self.covar["Ux_pV"] * sinTheta * cosν \
                - self.covar["Uy_pV"] * sinν * sinTheta
        if np.abs(Uz_pV) >= np.abs(self.covar["Uz_pV"]):
            self.covar["Uz_pV"] = Uz_pV
        self.covar["Ux_Q"] = self.covar["Ux_Q"] * cosTheta * cosν + self.covar["Uy_Q"] * cosTheta * sinν \
                             + self.covar["Uz_Q"] * sinTheta
        self.covar["Uy_Q"] = self.covar["Uy_Q"] * cosν - self.covar["Uy_Q"] * sinν
        self.covar["Uz_Q"] = self.covar["Uz_Q"] * cosTheta - self.covar["Ux_Q"] * sinTheta * cosν \
                             - self.covar["Uy_Q"] * sinν * sinTheta
        self.covar["Ux_Uz"] = self.covar["Ux_Uz"] * cosν * (
                cosTheta ** 2 - sinTheta ** 2) - 2 * self.covar["Ux_Uy"] * sinTheta * cosTheta * sinν * cosν \
                              + self.covar["Uy_Uz"] * sinν * (cosTheta ** 2 - sinTheta ** 2) \
                              - self.UxMSE * sinTheta * cosTheta * cosν ** 2 \
                              - self.UyMSE * sinTheta * cosTheta * sinν ** 2 + self.UzMSE * sinTheta * cosTheta
        self.covar["Uy_Uz"] = self.covar["Uy_Uz"] * cosTheta * cosν - self.covar["Ux_Uz"] * cosTheta * sinν \
                              - self.covar["Ux_Uy"] * sinTheta * (cosν ** 2 - sinν ** 2) \
                              + self.UxMSE * sinTheta * sinν * cosν - self.UyMSE * sinTheta * sinν * cosν
        self.covar["Uz_Sd"] = self.covar["Uz_Sd"] * cosTheta - self.covar["Ux_Sd"] * sinTheta * cosν \
                              - self.covar["Uy_Sd"] * sinν * sinTheta
        self.covar["Uxy_Uz"] = np.sqrt(self.covar["Ux_Uz"] ** 2 + self.covar["Uy_Uz"] ** 2)

    def webb_pearman_leuning(self, lamb, Tsa, pVavg, Uz_Ta, Uz_pV):
        """Webb, Pearman and Leuning Correction (WPL) for density.

        Args:
            lamb: initial latent heat estimate
            Tsa: Average air temperature
            pVavg: Average density of water in air
            Uz_Ta: Covariance Uz and Ta (vert wind component and air temp)
            Uz_pV: Covariance Uz and pV (vert wind component and water vapor density)

        Returns:
            Corrected Latent heat
        References:
            Webb, EK, Pearman, GI, and Leuning, R, (1980) Correction of the flux measurements for density effects due to
                heat and water vapour transfer. Quart J Roy Meteorol Soc. 106:85-100.
        Notes:
            The WPL-correction is large if the turbulent fluctuations are small relative to the mean concentration.
            For example, this is the case for carbon dioxide where corrections up to 50% are typical.
            For water vapour flux, the corrections are only a few percent because the effects of the Bowen ratio
            and the sensible heat flux balance each other.

        """

        # Webb, Pearman and Leuning Correction
        pCpTsa = self.p * self.Cp * Tsa
        pRatio = (1.0 + 1.6129 * (pVavg / self.pD))
        LE = lamb * pCpTsa * pRatio * (Uz_pV + (pVavg / Tsa) * Uz_Ta) / (pCpTsa + lamb * pRatio * pVavg * 0.07)

        return LE

    # @njit
    def calc_LnKh(self, mvolts):
        """Converts KH20 output to ln(KH20) for determination of vapor density

        Args:
            mvolts: output voltage of KH20 (mV)

        Returns:
            ln(mvolts)

        """
        return np.log(mvolts)

    def renamedf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames variables in input dataframe to accept

        Args:
            df: Raw high-frequency DataFrame with standard Campbell Scientific Header

        Returns:
            df: Dataframe with headers that will work with the script

        """
        return df.rename(columns={'T_SONIC': 'Ts',
                                  'TA_1_1_1': 'Ta',
                                  'amb_press': 'Pr',
                                  'PA':'Pr',
                                  'RH_1_1_1': 'Rh',
                                  't_hmp': 'Ta',
                                  'e_hmp': 'Ea',
                                  'kh': 'volt_KH20'
                                  })

    # @njit
    def despike(self, arr: np.ndarray, nstd: float = 4.5) -> np.ndarray:
        """Removes spikes from an array of values based on a specified deviation from the mean.

        Args:
            arr: array of values with spikes
            nstd: number of standard deviations from mean; default is 4.5

        Returns:
            Array of despiked values where spikes are replaced with interpolated values
        Notes:
            * Not windowed.
            * This method is fast but might be too agressive if the spikes are small relative to seasonal variability.
        """

        stdd = np.nanstd(arr) * nstd
        avg = np.nanmean(arr)
        avgdiff = stdd - np.abs(arr - avg)
        y = np.where(avgdiff >= 0, arr, np.NaN)
        nans, x = np.isnan(y), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    def despike_rolling_med(self, data, variable, window=1200, cutoff=3.5, drop_spikes=True):
        """

        Args:
            data:
            variable:
            window:
            cutoff:
            drop_spikes:

        Returns:

        """
        rollmed = data[variable].rolling(window=window, center=True).median().dropna()
        rollmedall = data[variable].rolling(window=window, center=True).median()
        # traindata = data.loc[rollmed.first_valid_index():rollmed.last_valid_index(),variable]
        traindata = data.loc[:, variable]

        rollmedall.loc[:rollmed.first_valid_index()] = data.loc[:rollmed.first_valid_index(), variable].median()
        rollmedall.loc[rollmed.last_valid_index():] = data.loc[rollmed.last_valid_index():, variable].median()

        mod_rlm = sm.RLM(np.array(rollmedall), np.array(traindata)).fit(maxiter=600, scale_est='mad', update_scale=True)

        spike_dates = rollmedall.index[np.where(np.abs(mod_rlm.resid / mod_rlm.scale) > cutoff)]
        spike_df = data.loc[spike_dates, variable].to_frame()

        if drop_spikes:
            data.loc[spike_dates, variable] = None

        return data[variable], spike_dates

    def despike_rolling(self, df, p, win=12, upper_threshold=5, lower_threshold=-1):
        """Removes spikes from an array of values based on a specified deviation from the rolling median.
        This is not fast

        Args:
            df: dataframe with values
            p: value field to despike
            win: size of window in data timesteps
            upper_threshold: max difference from rolling median allowed
            lower_threshold: min difference from rolling median allowed

        Returns:

        """
        # https://stackoverflow.com/questions/62692771/outlier-detection-based-on-the-moving-mean-in-python
        # Set threshold for difference with rolling median

        # Calculate rolling median
        df['rolling_temp'] = df[p].rolling(window=win).median()

        # Calculate difference
        df['diff'] = df[p] - df['rolling_temp']

        # Flag rows to be dropped as `1`
        df['drop_flag'] = np.where((df['diff'] > upper_threshold) | (df['diff'] < lower_threshold), 1, 0)

        # Drop flagged rows
        df[p + '_ds'] = df[p]
        df.loc[df['drop_flag'] == 1, p + '_ds'] = None
        df[p + '_ds'] = df[p + '_ds'].interpolate()
        df = df.drop(['rolling_temp', 'rolling_temp', 'diff', 'drop_flag'], axis=1)
        return df

    def get_lag(self, x, y):
        """Get cross-correlation of a signal against another signal.
        x = array of signal 1
        y = array of signal 2

        Notes
            From https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html#scipy.signal.correlation_lags
        """
        correlation = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        lag = lags[np.argmax(correlation)]
        return lag


    # @njit(parallel=True)
    def calc_Td_dewpoint(self, E):
        """Calculate Dew Point Temperature (K) from water vapor pressure

        Args:
            E: saturation vapor pressure (Pa)

        Returns:
            dew point for the given vapor pressure (K)

        References:
            From ITS-90 FORMULATIONS FOR VAPOR PRESSURE, FROSTPOINTTEMPERATURE, DEWPOINT TEMPERATURE,
            AND ENHANCEMENT FACTORS IN THE RANGE –100 TO +100 C by Bob Hardy see eq 4
            https://www.decatur.de/javascript/dew/resources/its90formulas.pdf

        Notes:
            Agrees with modified Wexler to 0.3 mK over the range of –100 to 100°C

        """

        c0 = 207.98233
        c1 = -20.156028
        c2 = 0.46778925
        c3 = -0.0000092288067

        d0 = 1
        d1 = -0.13319669
        d2 = 0.0056577518
        d3 = -0.000075172865
        lne = np.log(E)
        nom = (c0 + c1 * lne + c2 * lne ** 2 + c3 * lne ** 3)
        denom = (d0 + d1 * lne + d2 * lne ** 2 + d3 * lne ** 3)
        return nom / denom

    # @njit(parallel=True)
    def calc_Tf_frostpoint(self, E):
        """Calculate Frost Point Temperature (K) from water vapor pressure

        Args:
            E: saturation vapor pressure (Pa)

        Returns:
            frost point for the given vapor pressure (K)

        References:
            From ITS-90 FORMULATIONS FOR VAPOR PRESSURE, FROSTPOINTTEMPERATURE, DEWPOINT TEMPERATURE,
            AND ENHANCEMENT FACTORS IN THE RANGE –100 TO +100 C by Bob Hardy see eq 5
            https://www.decatur.de/javascript/dew/resources/its90formulas.pdf

        Notes:
            Agrees with modified Wexler to 0.1 mK over the range of –150 to 0.1°C


        """

        c0 = 207.98233
        c1 = -20.156028
        c2 = 0.46778925
        c3 = -0.0000092288067

        d0 = 1
        d1 = -0.13319669
        d2 = 0.0056577518
        d3 = -0.000075172865

        lne = np.log(E)
        nom = (c0 + c1 * lne + c2 * lne ** 2)
        denom = (d0 + d1 * lne + d2 * lne ** 2 + d3 * lne ** 3)
        return nom / denom

    # @njit(parallel=True)
    def calc_E_vapor_pressure(self, pV, T):
        """Ideal Gas Law to calculate vapor pressure from water vapor density and temperature;

        Args:
            pV: Density of water vapor in air (kg/m3)
            T: Sonic Temperature (K)

        Returns:
            E Actual Vapor Pressure (Pa)

        Notes:
            Constant Rv is gas constant of Water Vapor (J/(kg K))

        Examples:
            >>> fluxcalc = CalcFlux()
            >>> print(fluxcalc.calc_E_vapor_pressure(3.4,290.2))
            455362.68679999997

        """

        e = pV * T * self.Rv
        return e

    # @njit(parallel=True)
    def calc_Q_specific_humidity(self, P, e):
        """Calculate Specific Humidity from pressure and vapor pressure

        Args:
            P: Air pressure (Pa)
            e: Actual Vapor Pressure (Pa)

        Returns:
            Specific Humidity (unitless)

        References:
            Bolton 1980

        Notes:
            Specific humidity is the ratio of the mass of water vapor to the mass of moist air

        Examples:
            >>> fluxcalc = CalcFlux()
            >>> print(fluxcalc.calc_Q_specific_humidity(np.array([4003.6,4002.1]),np.array([717,710])))
            [0.11948162 0.11827882]

            >>> fluxcalc = CalcFlux()
            >>> print(fluxcalc.calc_Q_specific_humidity(4003.6,717))
            0.11948162313727738

        """

        # molar mass of water vapor/ molar mass of dry air
        gamma = 0.622
        q = (gamma * e) / (P - 0.378 * e)
        return q

    def calc_tc_air_temp_sonic(self, Ts, pV, P_atm):
        """Air temperature from sonic temperature, water vapor density, and atmospheric pressure from Campbell Scientific EasyFLux

        Args:
            Ts: Sonic Temperature (K)
            pV: h2O density (kg m-3)
            P_atm: Pressure (atm)

        Returns:
            Ta Air Temperature

        References:
            Wallace and Hobbs 2006
        """

        T_c1 = P_atm + (2 * self.Rv - 3.040446 * self.Rd) * pV * Ts
        T_c2 = P_atm * P_atm + (1.040446 * self.Rd * pV * Ts) * (
                    1.040446 * self.Rd * pV * Ts) + 1.696000 * self.Rd * pV * P_atm * Ts
        T_c3 = 2 * pV * ((self.Rv - 1.944223 * self.Rd) + (self.Rv - self.Rd) * (
                    self.Rv - 2.040446 * self.Rd) * pV * Ts / P_atm)

        return (T_c1 - np.sqrt(T_c2)) / T_c3

    # @njit
    def calc_Tsa_air_temp_sonic(self, Ts, q):
        """Calculate air temperature from sonic temperature and specific humidity

        Args:
            Ts: Sonic Temperature (K)
            q: Specific Humidity (unitless)

        Returns:
            Tsa (air temperature derived from sonic temperature, K)

        References:
            Schotanus et al. (1983) doi:10.1007/BF00164332
            Also see Kaimal and Gaynor (1991) doi:10.1007/BF00119215

        Examples:
            >>> fluxcalc = CalcFlux()
            >>> print(fluxcalc.calc_Tsa_air_temp_sonic(291.5,0.8))
            207.03125

        """

        Tsa = Ts / (1 + 0.51 * q)
        return Tsa

    def calc_L(self, Ust, Tsa, Uz_Ta):
        """Calculates the Monin-Obukhov length

        Args:
            Ust: friction velocity (Ustar); a measure of surface stress (m/s)
            Tsa: virtual temperature (K)
            Uz_Ta: kinematic virtual temperature flux

        Returns:
            Monin-Obukhov length; demoninator of stability parameter z/L

        Examples:
            >>> fluxcalc = CalcFlux()
            >>> print(fluxcalc.calc_L(1.2, 292.2, 1.5))
            -85.78348623853208
        """
        return (-1 * (Ust ** 3) * Tsa) / (self.g * self.von_karman * Uz_Ta)

    # @numba.njit#(forceobj=True)
    def calc_Tsa(self, Ts, P, pV, Rv=461.51):
        """
        Calculate the average sonic temperature
        :param Ts:
        :param P:
        :param pV:
        :param Rv:
        :return:
        """
        E = pV * self.Rv * Ts
        return -0.01645278052 * (
                -500 * P - 189 * E + np.sqrt(250000 * P ** 2 + 128220 * E * P + 35721 * E ** 2)) / pV / Rv

    # @numba.jit(forceobj=True)
    def calc_AlphX(self, L):
        if (self.UHeight / L) <= 0:
            alph = 0.925
            X = 0.085
        else:
            alph = 1
            X = 2 - 1.915 / (1 + 0.5 * self.UHeight / L)
        return alph, X

    # @numba.njit#(forceobj=True)
    def tetens(self, t, a=0.611, b=17.502, c=240.97):
        """Magnus Tetens formula for computing the saturation vapor pressure of water from temperature; eq. 3.8

        Args:
            t: Temperature (C)
            a: constant a (kPa)
            b: constant b (dimensionless)
            c: constant c (C)

        Returns:
            saturation vapor pressure (kPa)
        """

        return a * np.exp((b * t) / (t + c))

    # @numba.jit(forceobj=True)
    def calc_Es_sat_vapor_pressure(self, T):
        """Saturation Vapor Pressure Equation modified by Hardy from Wexler

        Args:
            T: Water temperature in Kelvin

        Returns:
            Saturation Vapor Pressure (Pa)

        References:
            From ITS-90 FORMULATIONS FOR VAPOR PRESSURE, FROST POINT TEMPERATURE, DEWPOINT TEMPERATURE,
            AND ENHANCEMENT FACTORS IN THE RANGE –100 TO +100 C by Bob Hardy see eq 2
            The Proceedings of the Third International Symposium on Humidity & Moisture,
            Teddington, London, England, April 1998
            https://www.decatur.de/javascript/dew/resources/its90formulas.pdf

        Notes:
            within 0.05 ppm from -100 to 100 deg C
        """

        g0 = -2836.5744
        g1 = -6028.076559
        g2 = 19.54263612
        g3 = -0.02737830188
        g4 = 0.000016261698
        g5 = 0.00000000070229056
        g6 = -0.00000000000018680009
        g7 = 2.7150305

        return np.exp(
            g0 * T ** (-2) + g1 * T ** (-1) + g2 + g3 * T + g4 * T ** 2 + g5 * T ** 3 + g6 * T ** 4 + g7 * np.log(T))

    # @njit(parallel=True)
    def calc_cov(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate covariance between two variables;

        Args:
            p1: array of variable 1
            p2: array of variable 2 (must be same dimensions as p1)

        Returns:
            covariance between p1 and p2

        Notes:
            When used with njit(parallel=True) performs about order of mag. faster than numpy cov or pandas cov
            when len(p1) > 10,000;
        """

        sumproduct = np.sum(p1 * p2)
        return (sumproduct - (np.sum(p1) * np.sum(p2)) / len(p1)) / (len(p1) - 1)

    # @numba.njit#(forceobj=True)
    def calc_MSE(self, y: np.ndarray) -> float:
        """Calculate mean standard error

        Args:
            y: numpy array of variable

        Returns:
            Mean standard error of an array
        """

        return np.mean((y - np.mean(y)) ** 2)

    def convert_KtoC(self, T):
        """Convert Kelvin to Celcius

        Args:
            T: Temperature in Kelvin

        Returns:
            Temperature in Celcius
        """

        return T - 273.16

    def convert_CtoK(self, T):
        """Convert Celcius to Kelvin

        Args:
            T: Temperature in Celcius degrees

        Returns:
            Temperature in Kelvin
        """

        return T + 273.16

    def correct_KH20(self, Uz_Ta, P, T):
        """Calculates an additive correction for the KH20 due to cross sensitivity between H20 and 02 molecules.

        Args:
            Uz_Ta: Covariance of Vertical Wind Component and Air Temperature (m*K/s)
            P: Air Pressure (Pa)
            T: Air Temperature (K)

        Returns:
            KH20 Oxygen Correction
        Notes
            * Kw = Extinction Coefficient of Water (m^3/[g*cm]) -instrument calibration
            * Ko = Extinction Coefficient of Oxygen (m^3/[g*cm]) -derived experimentally
        """
        return ((self.Co * self.Mo * P) / (self.Ru * T ** 2)) * (self.Ko / self.Kw) * Uz_Ta

    def correct_spectral(self, B, alpha, varib):
        B_alpha = B ** alpha
        V_alpha = varib ** alpha
        return (B_alpha / (B_alpha + 1)) * (B_alpha / (B_alpha + V_alpha)) * (1 / (V_alpha + 1))

    def get_Watts_to_H2O_conversion_factor(self, temperature, day_fraction):
        to_inches = 25.4
        return (self.calc_water_density(temperature) * 86.4 * day_fraction) / (
                self.calc_latent_heat_of_vaporization(temperature) * to_inches)

    def calc_water_density(self, temperature):
        """Calculate the density of water (kg/m3) for a given temperature (C)"""
        d1 = -3.983035  # °C
        d2 = 301.797  # °C
        d3 = 522528.9  # °C2
        d4 = 69.34881  # °C
        d5 = 999.97495  # kg/m3
        return d5 * (1 - (temperature + d1) ** 2 * (temperature + d2) / (d3 * (temperature + d4)))  # 'kg/m^3

    def calc_latent_heat_of_vaporization(self, temperature):
        """From Rogers and Yau (1989) A Short Course in Cloud Physics
        https://en.wikipedia.org/wiki/Latent_heat
        :param temperature: temperature in degrees C
        :return: Specific Latent Heat of Condensation of Water (J/kg)
        """
        if temperature >= 0:
            l0 = 2500800
            l1 = -2360
            l2 = 1.6
            l3 = -0.06
        else:
            # these parameters are for sublimation from ice
            l0 = 2834100
            l1 = -290
            l2 = -4
            l3 = 0
        return l0 + l1 * temperature + l2 * temperature ** 2 + l3 * temperature ** 3  # 'J/kg

    @njit(parallel=True)
    def shadow_correction(self, Ux, Uy, Uz):
        """Correction for flow distortion of CSAT sonic anemometer from Horst and others (2015) based on work by Kaimal

        :param Ux: Longitudinal component of the wind velocity (m s-1); aka u
        :param Uy: Lateral component of the wind velocity (m s-1); aka v
        :param Uz: Vertical component of the wind velocity (m s-1); aka w
        :return: corrected wind components
        """

        # Rotation Matrix to Align with Path Coordinate System of Transducers
        h = [0.25, 0.4330127018922193, 0.8660254037844386,
             -0.5, 0.0, 0.8660254037844386,
             0.25, -0.4330127018922193, 0.8660254037844386]

        # Inverse of the Rotation Matrix
        hinv = [0.6666666666666666, -1.3333333333333333, 0.6666666666666666,
                1.1547005383792517, 0.0, -1.1547005383792517,
                0.38490017945975047, 0.38490017945975047, 0.38490017945975047]

        iteration = 0

        while iteration < 4:
            Uxh = h[0] * Ux + h[1] * Uy + h[2] * Uz
            Uyh = h[3] * Ux + h[4] * Uy + h[5] * Uz
            Uzh = h[6] * Ux + h[7] * Uy + h[8] * Uz

            scalar = np.sqrt(Ux ** 2. + Uy ** 2. + Uz ** 2.)

            Theta1 = np.arccos(np.abs(h[0] * Ux + h[1] * Uy + h[2] * Uz) / scalar)
            Theta2 = np.arccos(np.abs(h[3] * Ux + h[4] * Uy + h[5] * Uz) / scalar)
            Theta3 = np.arccos(np.abs(h[6] * Ux + h[7] * Uy + h[8] * Uz) / scalar)

            #  Adjustment Factors for Each Component
            # Adjust for the Shadowing Effects

            Uxa = Uxh / (0.84 + 0.16 * np.sin(Theta1))
            Uya = Uyh / (0.84 + 0.16 * np.sin(Theta2))
            Uza = Uzh / (0.84 + 0.16 * np.sin(Theta3))

            # Transform the Winds Components Back to the CSAT Coordinate System.
            # These are the Corrected Velocities.

            Uxc = hinv[0] * Uxa + hinv[1] * Uya + hinv[2] * Uza
            Uyc = hinv[3] * Uxa + hinv[4] * Uya + hinv[5] * Uza
            Uzc = hinv[6] * Uxa + hinv[7] * Uya + hinv[8] * Uza

            Ux = Uxc
            Uy = Uyc
            Uz = Uzc

            iteration += 1

        return Uxc, Uyc, Uzc

    # Calculated Weather Parameters
    # @numba.jit
    def calculated_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate weather parameters from raw data.

        Args:
            df: DataFrame containing Ea, Ts, and Pr

        Returns:
            Adds new fields to dataframe, including pV, Tsa, Q, and Sd
        """
        df['pV'] = self.calc_pV(df['Ea'], df['Ts'])
        df['Tsa'] = self.calc_Tsa(df['Ts'], df['Pr'], df['pV'])
        df['E'] = self.calc_E(df['pV'], df['Tsa'])
        df['Q'] = self.calc_Q(df['Pr'], df['E'])
        df['Sd'] = self.calc_Q(df['Pr'], self.calc_Es(df['Tsa'])) - df['Q']
        return df

    def calculated_parameters_irga(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate weather parameters from raw data.

        Args:
            df: DataFrame containing Ta, Ts, and Pr

        Returns:
            Adds new fields to dataframe, including pV, Tsa, Q, and Sd
        """
        # convert pV to g/m-3
        df['Ts_K'] = self.convert_CtoK(df['Ts'])
        df = self.despike_rolling(df, 'pV', win=12, upper_threshold=5, lower_threshold=-1)
        df['E'] = self.calc_E_vapor_pressure(df['pV_ds'] * 0.001, df['Ts_K'])
        # convert air pressure from kPa to Pa
        df['Q'] = self.calc_Q_specific_humidity(df['Pr'] * 1000., df['E'])
        df['Tsa'] = self.calc_Tsa_air_temp_sonic(df['Ts_K'], df['Q'])
        df['Es'] = self.calc_Es_sat_vapor_pressure(df['Tsa'])

        df['Sd'] = self.calc_Q(df['Pr'], self.calc_Es(df['Tsa'])) - df['Q']
        return df

    # @numba.njit#(forceobj=True)
    def calc_pV_water_vapor_density(self, Ea, Ts):
        """

        Args:
            Ea: Actual vapor pressure (Pa)
            Ts: Sonic Temperature (K)

        Returns:
            Water Vapor Density
        """
        return (Ea * 1000.0) / (self.Rv * Ts)

    def calc_max_covariance(self, x: np.ndarray, y: np.ndarray, lag: int = 10) -> [(int, float), (int, float),
                                                                                   (int, float), dict]:
        """Shift Arrays in Both Directions and Calculate Covariances for Each Lag.
        This Will Account for Longitudinal Separation of Sensors or Any Synchronization Errors.

        Args:
            x: array 1
            y: array 2 (must be same dimensions as array1
            lag: number of lags to examine; defaults to 10

        Returns:
            three tuples and a dictionary; abscov = (lag value of highest absolute cov, highest absolute cov),
            maxcov = (lag value of max positive cov, max positive cov),
            mincov = (lag value of max negative cov, max negative cov),
            xy = dictionary of all lags and their corresponding covariance values
        """

        xy = {}

        for i in range(0, lag + 1):
            if i == 0:
                xy[0] = np.round(np.cov(x, y)[0][1], 8)
                x_y = xy[0]
            else:
                # covariance for positive lags
                xy[i] = np.round(np.cov(x[i:], y[:-1 * i])[0][1], 8)
                # covariance for negative lags
                xy[-i] = np.round(np.cov(x[:-1 * i], x[i:])[0][1], 8)

        # convert dictionary to arrays
        keys = np.array(list(xy.keys()))
        vals = np.array(list(xy.values()))

        # get index and value for maximum positive covariance
        valmax = np.max(vals)
        maxlagindex = np.where(vals == valmax)[0][0]
        maxlag = keys[maxlagindex]
        maxcov = (maxlag, valmax)

        # get index and value for get maximum negative covariance
        valmin = np.min(vals)
        minlagindex = np.where(vals == valmin)[0][0]
        minlag = keys[minlagindex]
        mincov = (minlag, valmin)

        # get index and value for get maximum absolute covariance
        absmax = np.max(np.abs(vals))
        abslagindex = np.where(np.abs(vals) == np.abs(absmax))[0][0]
        absmaxlag = keys[abslagindex]
        abscov = (absmaxlag, absmax)

        return abscov, maxcov, mincov, xy

    def calc_max_covariance_df(self, df: pd.DataFrame, colx: str, coly: str, lags: int = 10) -> [float, int]:
        """
        Find maximum covariance between two variables
        :param df: Pandas DataFrame containing the data
        :param colx: DataFrame column with x variable
        :param coly: DataFrame column with y variable
        :param lags: number of lags to search over; default is 10; larger number requires more time
        :return: maximum covariance between two variables, lag number of max covariance
        """
        dfcov = []
        for i in np.arange(-1 * lags, lags):
            df[f"{coly}_{i}"] = df[coly].shift(i)
            dfcov.append(df[[colx, f"{coly}_{i}"]].cov().loc[colx, f"{coly}_{i}"])
            # print(i,df[[colx, f"{coly}_{i}"]].cov().loc[colx, f"{coly}_{i}"])
            df = df.drop([f"{coly}_{i}"], axis=1)

        abscov = np.abs(dfcov)
        maxabscov = np.max(abscov)
        try:
            maxlagindex = np.where(abscov == maxabscov)[0][0]
            lagno = maxlagindex - lags
            maxcov = dfcov[maxlagindex]
        except IndexError:
            lagno = 0
            maxcov = dfcov[10]
        return maxcov, lagno

    # @numba.njit#(forceobj=True)
    def coord_rotation(self, df: pd.DataFrame, Ux: str = 'Ux', Uy: str = 'Uy', Uz: str = 'Uz') -> pd.DataFrame:
        """Traditional Coordinate Rotation; The first correction is the rotation of the coordinate system around the
        z-axis into the mean wind.

        Args:
            df: Dataframe containing the wind velocity components
            Ux: Longitudinal component of the wind velocity (m s-1); aka u
            Uy: Lateral component of the wind velocity (m s-1); aka v
            Uz: Vertical component of the wind velocity (m s-1); aka w

        Returns:

        References:
            Based on work by Tanner and Thurtell (1969) and Hyson et al. (1977).
            From Kaimal and Finnigan 1994

        """

        xmean = df[Ux].mean()
        ymean = df[Uy].mean()
        zmean = df[Uz].mean()
        Uxy = np.sqrt(xmean ** 2 + ymean ** 2)
        Uxyz = np.sqrt(xmean ** 2 + ymean ** 2 + zmean ** 2)
        cosν = xmean / Uxy
        sinν = ymean / Uxy
        sinTheta = zmean / Uxyz
        cosTheta = Uxy / Uxyz
        return cosν, sinν, sinTheta, cosTheta, Uxy, Uxyz

    def coordinate_rotation(self, Ux, Uy, Uz):
        """Traditional Coordinate Rotation

        Args:
            Ux: Longitudinal component of the wind velocity (m s-1); aka u
            Uy: Lateral component of the wind velocity (m s-1); aka v
            Uz: Vertical component of the wind velocity (m s-1); aka w

        Returns:

        """

        xmean = np.mean(Ux)
        ymean = np.mean(Uy)
        zmean = np.mean(Uz)
        Uxy = np.sqrt(xmean ** 2 + ymean ** 2)
        Uxyz = np.sqrt(xmean ** 2 + ymean ** 2 + zmean ** 2)
        cosν = xmean / Uxy
        sinν = ymean / Uxy
        sinTheta = zmean / Uxyz
        cosTheta = Uxy / Uxyz
        return cosν, sinν, sinTheta, cosTheta, Uxy, Uxyz

    def dayfrac(self, df):
        return (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(1, unit='D')

    def calc_covar(self, Ux, Uy, Uz, Ts, Q, pV=None):
        """Calculate standard covariances of primary variables

        Args:
            Ux: Longitudinal component of the wind velocity (m s-1); aka u
            Uy: Lateral component of the wind velocity (m s-1); aka v
            Uz: Vertical component of the wind velocity (m s-1); aka w
            Ts: Sonic Temperature
            Q: Humidity
            pV: Vapor Density

        Returns:
            Saves resulting covariance to the `covar` dictionary object; ex self.covar['Ux_Ux']
        """

        self.covar['Ts_Ts'] = self.calc_cov(Ts, Ts)
        self.covar['Ux_Ux'] = self.calc_cov(Ux, Ux)
        self.covar['Uy_Uy'] = self.calc_cov(Uy, Uy)
        self.covar['Uz_Uy'] = self.calc_cov(Uz, Uz)
        self.covar['Q_Q'] = self.calc_cov(Q, Q)
        if pV:
            self.covar['pV_pV'] = self.calc_cov(pV, pV)
