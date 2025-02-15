# Original scripts in Fortran by Lawrence Hipps USU
# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import pandas as pd
import numpy as np
from scipy import signal
import statsmodels.api as sm


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

        self.Cp = None
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
        self.stdvals = {} # Dictionary of the standard deviations
        self.errvals = {} # Dictionary of standard errors

        self.cosv = None
        self.sinv = None
        self.sinTheta = None
        self.cosTheta = None

        # List of Variables for despiking
        self.despikefields = ['Ux', 'Uy', 'Uz', 'Ts', 'volt_KH20', 'Pr', 'Rh','pV']

        self.wind_compass = None
        self.pathlen = None
        self.df = None

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
                df[col] = self.despike_quart_filter(df[col])

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

        self.covar["Ts-Q"] = self.calc_max_covariance(df['Ts'], df['Q'], self.lag)[0][1]

        # Traditional Coordinate Rotation
        cosv, sinv, sinTheta, cosTheta, Uxy, Uxyz = self.coord_rotation(df)

        df = self.rotate_velocity_values(df, 'Ux', 'Uy', 'Uz')

        # Find the Mean Squared Error of Velocity Components and Humidity
        self.UxMSE = self.calc_MSE(df['Ux'])
        self.UyMSE = self.calc_MSE(df['Uy'])
        self.UzMSE = self.calc_MSE(df['Uz'])
        self.QMSE = self.calc_MSE(df['Q'])

        # Correct Covariances for Coordinate Rotation
        self.covar_coord_rot_correction(cosv, sinv, sinTheta, cosTheta)

        Ustr = np.sqrt(self.covar["Uxy-Uz"])

        # Find Average Air Temperature From Average Sonic Temperature
        Tsa = self.calc_Tsa(df['Ts'].mean(), df['Pr'].mean(), df['pV'].mean())

        # Calculate the Latent Heat of Vaporization (eq. 2.57 in Foken)
        lamb = (2500800 - 2366.8 * (self.convert_KtoC(Tsa)))

        # Determine Vertical Wind and Water Vapor Density Covariance
        #Uz_pV = (self.covar["Uz-pV"] / XKw) / 1000

        # Calculate the Correct Average Values of Some Key Parameters
        self.Cp = self.Cpd * (1 + 0.84 * df['Q'].mean())
        self.pD = (df['Pr'].mean() - df['E'].mean()) / (self.Rd * Tsa)
        self.p = self.pD + df['pV'].mean()

        # Calculate Variance of Air Temperature From Variance of Sonic Temperature
        StDevTa = np.sqrt(self.covar["Ts-Ts"]
                          - 1.02 * df['Ts'].mean() * self.covar["Ts-Q"]
                          - 0.2601 * self.QMSE * df['Ts'].mean() ** 2)
        Uz_Ta = self.covar["Uz-Ts"] - 0.07 * lamb * self.covar["Uz-pV"] / (self.p * self.Cp)

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td_dewpoint(df['E'].mean())
        D = self.calc_Es(Tsa) - df['E'].mean()
        S = (self.calc_Q(df['Pr'].mean(),
                                           self.calc_Es(Tsa + 1)) - self.calc_Q(df['Pr'].mean(),
                                                                                self.calc_Es(Tsa - 1))) / 2

        # Determine Wind Direction
        Ux_avg = np.mean(df['Ux'].to_numpy())
        Uy_avg = np.mean(df['Uy'].to_numpy())
        Uz_avg = np.mean(df['Uz'].to_numpy())

        pathlen, direction = self.determine_wind_dir(Ux_avg, Uy_avg)

        # Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df['Uz'].std()
        UMean = Ux_avg * cosTheta * cosv + Uy_avg * cosTheta * sinv + Uz_avg * sinTheta

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
        self.covar["Uz-pV"] /= KH20
        self.covar["Uxy_Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy_Uz"])
        self.covar["Uz_Sd"] /= KH20
        exchange = ((self.p * self.Cp) / (S + self.Cp / lamb)) * self.covar["Uz_Sd"]

        # KH20 Oxygen Correction
        self.covar["Uz-pV"] += self.correct_KH20(Uz_Ta, df['Pr'].mean(), Tsa)

        # Calculate New H and LE Values
        H = self.p * self.Cp * Uz_Ta
        lambdaE = lamb * self.covar["Uz-pV"]

        # Webb, Pearman and Leuning Correction
        pVavg = np.mean(df['pV'].to_numpy())
        lambdaE = self.webb_pearman_leuning(lamb, Tsa, pVavg, Uz_Ta, self.covar["Uz-pV"])

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

    def run_irga(self, df: pd.DataFrame) -> pd.Series:
        """Runs through complete processing of eddy covariance dataset.

        Args:
            df: dataframe Weather Parameters for the Eddy Covariance Method;
            must be time-indexed and include Ux, Uy, Uz, Pr, Ea, and pV or LnKH

        Returns:
            A series of values aggregated over the length of the provided dataframe

        """
        df = self.renamedf(df)

        for col in self.despikefields:
            if col in df.columns:
                df[col + "_ro"] = self.despike_med_mod(df[col])

        # Convert Sonic and Air Temperatures from Degrees C to Kelvin
        df.loc[:, 'Ts'] = self.convert_CtoK(df['Ts_ro'].to_numpy())
        df.loc[:, 'Ta'] = self.convert_CtoK(df['Ta'].to_numpy())

        # Remove shadow effects of the CSAT (this is also done by the CSAT Firmware)
        df['Ux'], df['Uy'], df['Uz'] = self.shadow_correction(df['Ux_ro'].to_numpy(),
                                                                  df['Uy_ro'].to_numpy(),
                                                                  df['Uz_ro'].to_numpy())
        #print(df[['Ux','Uy','Uz']])
        self.avgvals = df.mean().to_dict()

        # Calculate Sums and Means of Parameter Arrays
        # convert air pressure from kPa to Pa
        df['Pr'] = df['Pr_ro'] * 1000.
        # convert pV to g/m-3
        df['pV'] = df['pV_ro'] * 0.001

        df['E'] = self.calc_E(df['pV'], df['Ts'])
        # convert air pressure from kPa to Pa
        df['Q'] = self.calc_Q(df['Pr'], df['E'])
        df['Tsa'] = self.calc_Tsa(df['Ts'], df['Q'])
        #df['Tsa2'] = self.calc_tc_air_temp_sonic(df['Ts'], df['pV'], df['Pr'])
        df['Es'] = self.calc_Es(df['Tsa'])
        df['Sd'] = self.calc_Q(df['Pr'], self.calc_Es(df['Tsa'])) - df['Q']

        # Calculate Covariances (Maximum Furthest From Zero With Sign in Lag Period)
        self.calc_covar(df['Ux'].to_numpy(),
                            df['Uy'].to_numpy(),
                            df['Uz'].to_numpy(),
                            df['Ts'].to_numpy(),
                            df['Q'].to_numpy(),
                            df['pV'].to_numpy())

        # Calculate max variance to close separation between sensors
        velocities = {"Ux": df["Ux"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                      "Uy": df["Uy"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                      "Uz": df["Uz"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy()}

        covariance_variables = {"Ux": df["Ux"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "Uy": df["Uy"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "Uz": df["Uz"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "Ts": df["Ts"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "Tsa": df["Tsa"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "pV": df["pV"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "Q": df["Q"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy(),
                                "Sd": df["Sd"].interpolate().fillna(method='bfill').fillna(method='ffill').to_numpy()}

        # This iterates through the velocities and calculates the maximum covariance between
        # the velocity and the other variables
        for ik, iv in velocities.items():
            for jk, jv in covariance_variables.items():
                try:
                    self.covar[f"{ik}-{jk}"] = self.calc_max_covariance(iv, jv)[0][1]
                except IndexError:
                    print(f"index error {ik}-{jk}")
                    self.covar[f"{ik}-{jk}"] = self.calc_cov(iv, jv)

        try:
            self.covar["Ts-Q"] = self.calc_max_covariance(df['Ts'].interpolate().fillna(method='bfill').fillna(method='ffill'),
                                                          df['Q'].interpolate().fillna(method='bfill').fillna(method='ffill'),
                                                          self.lag)[0][1]
        except IndexError:
            self.covar["Ts-Q"] = self.calc_cov(iv, jv)

        # Traditional Coordinate Rotation
        cosv, sinv, sinTheta, cosTheta, Uxy, Uxy_Uz = self.coord_rotation(df)

        df = self.rotate_velocity_values(df, 'Ux', 'Uy', 'Uz')

        # Find the Mean Squared Error of Velocity Components and Humidity
        for varib in ['Ux', 'Uy', 'Uz', 'Q', 'Ts', 'Tsa']:
            self.errvals[varib] = self.calc_MSE(df[varib].to_numpy())

        # Correct Covariances for Coordinate Rotation
        self.covar_coord_rot_correction(cosv, sinv, sinTheta, cosTheta)

        Ustr = np.sqrt(self.covar["Uxy-Uz"])

        # Find Average Air Temperature From Average Sonic Temperature
        Tsa = self.calc_Tsa_sonic_temp(df['Ts'].mean(), df['Pr'].mean(), df['pV'].mean())

        # Calculate the Latent Heat of Vaporization (eq. 2.57 in Foken)
        lamb = (2500800 - 2366.8 * (self.convert_KtoC(Tsa)))

        StDevTa = np.sqrt(np.abs(
            self.covar["Ts-Ts"] - 1.02 * df['Ts'].mean() * self.covar["Ts-Q"] - 0.2601 * self.errvals['Q'] * df[
                'Ts'].mean() ** 2))

        # Calculate the Correct Average Values of Some Key Parameters
        self.Cp = self.Cpd * (1 + 0.84 * df['Q'].mean())
        self.pD = (df['Pr'].mean() - df['E'].mean()) / (self.Rd * Tsa)
        self.p = self.pD + df['pV'].mean()

        # Calculate Variance of Air Temperature From Variance of Sonic Temperature
        StDevTa = np.sqrt(np.abs(self.covar["Ts-Ts"]
                                 - 1.02 * df['Ts'].mean() * self.covar["Ts-Q"]
                                 - 0.2601 * self.errvals['Q'] * df['Ts'].mean() ** 2))

        Uz_Ta = self.covar["Uz-Ts"] - 0.07 * lamb * self.covar["Uz-pV"] / (self.p * self.Cp)

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td_dewpoint(df['E'].mean())
        D = self.calc_Es(Tsa) - df['E'].mean()
        S = (self.calc_Q(df['Pr'].mean(),
                             self.calc_Es(Tsa + 1)) - self.calc_Q(df['Pr'].mean(),
                                                                          self.calc_Es(Tsa - 1))) / 2

        # Determine Wind Direction
        Ux_avg = np.mean(df['Uxr'].to_numpy())
        Uy_avg = np.mean(df['Uyr'].to_numpy())
        Uz_avg = np.mean(df['Uzr'].to_numpy())

        pathlen, direction = self.determine_wind_dir(Ux_avg, Uy_avg)

        # Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df['Uz'].std()
        UMean = Ux_avg * cosTheta * cosv + Uy_avg * cosTheta * sinv + Uz_avg * sinTheta

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
        self.covar["Uxy-Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy-Uz"])
    
        # Recalculate L With New Uᕽ and Uz_Ta, and Calculate High Frequency Corrections
        L = self.calc_L(Ustr, Tsa, Uz_Ta / Ts)
        alpha, X = self.calc_AlphX(L)
        Ts = self.correct_spectral(B, alpha, _Ts)
        KH20 = self.correct_spectral(B, alpha, _KH20)
    
        # Correct the Covariance Values
        Uz_Ta /= Ts
        self.covar["Uz-pV"] /= KH20
        self.covar["Uxy-Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy-Uz"])
        self.covar["Uz-Sd"] /= KH20
        exchange = ((self.p * self.Cp) / (S + self.Cp / lamb)) * self.covar["Uz-Sd"]
    
        # KH20 Oxygen Correction
        self.covar["Uz-pV"] += self.correct_KH20(Uz_Ta, df['Pr'].mean(), Tsa)
    
        # Calculate New H and LE Values
        H = self.p * self.Cp * Uz_Ta
        lambdaE = lamb * self.covar["Uz-pV"]
    
        # Webb, Pearman and Leuning Correction
        pVavg = np.mean(df['pV'].to_numpy())
        lambdaE = self.webb_pearman_leuning(lamb, Tsa, pVavg, Uz_Ta, self.covar["Uz-pV"])
    
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

    def determine_wind_dir(self, uxavg=None, uyavg=None, update_existing_vel=False):
        # Determine Wind Direction

        if uxavg:
            if update_existing_vel:
                self.avgvals['Ux'] = uxavg
        else:
            if 'Ux' in self.avgvals.keys():
                uxavg = self.avgvals['Ux']
            else:
                print('Please calculate wind velocity averages')
        if uyavg:
            if update_existing_vel:
                self.avgvals['Uy'] = uyavg
        else:
            if 'Uy' in self.avgvals.keys():
                uyavg = self.avgvals['Uy']
            else:
                print('Please calculate wind velocity averages')

        if uyavg and uxavg:
            self.v = np.sqrt(uxavg ** 2 + uyavg ** 2)
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

            self.wind_compass = wind_compass
            # Calculate the Lateral Separation Distance Projected Into the Mean Wind Direction
            self.pathlen = self.PathDist_U * np.abs(np.sin((np.pi / 180) * wind_compass))
            return self.pathlen, self.wind_compass

    def covar_coord_rot_correction(self, cosν=None, sinv=None, sinTheta=None, cosTheta=None):
        """Correct Covariances for Coordinate Rotation

        Args:
            cosν:
            sinv:
            sinTheta:
            cosTheta:

        Returns:

        """

        if cosTheta is None:
            cosν = self.cosv
            cosTheta = self.cosTheta
            sinv = self.sinv
            sinTheta = self.sinTheta

        #
        Uz_Ts = self.covar["Uz-Tsa"] * cosTheta - self.covar["Ux-Tsa"] * sinTheta * cosν \
                - self.covar["Uy-Tsa"] * sinTheta * sinv
        if np.abs(Uz_Ts) >= np.abs(self.covar["Uz-Tsa"]):
            self.covar["Uz-Tsa"] = Uz_Ts

        Uz_pV = self.covar["Uz-pV"] * cosTheta - self.covar["Ux-pV"] * sinTheta * cosν \
                - self.covar["Uy-pV"] * sinv * sinTheta
        if np.abs(Uz_pV) >= np.abs(self.covar["Uz-pV"]):
            self.covar["Uz-pV"] = Uz_pV
        self.covar["Ux-Q"] = self.covar["Ux-Q"] * cosTheta * cosν + self.covar["Uy-Q"] * cosTheta * sinv \
                             + self.covar["Uz-Q"] * sinTheta
        self.covar["Uy-Q"] = self.covar["Uy-Q"] * cosν - self.covar["Uy-Q"] * sinv
        self.covar["Uz-Q"] = self.covar["Uz-Q"] * cosTheta - self.covar["Ux-Q"] * sinTheta * cosν \
                             - self.covar["Uy-Q"] * sinv * sinTheta
        self.covar["Ux-Uz"] = self.covar["Ux-Uz"] * cosν * (
                cosTheta ** 2 - sinTheta ** 2) - 2 * self.covar["Ux-Uy"] * sinTheta * cosTheta * sinv * cosν \
                              + self.covar["Uy-Uz"] * sinv * (cosTheta ** 2 - sinTheta ** 2) \
                              - self.errvals['Ux'] * sinTheta * cosTheta * cosν ** 2 \
                              - self.errvals['Uy'] * sinTheta * cosTheta * sinv ** 2 + self.errvals['Uz'] * sinTheta * cosTheta
        self.covar["Uy-Uz"] = self.covar["Uy-Uz"] * cosTheta * cosν - self.covar["Ux-Uz"] * cosTheta * sinv \
                              - self.covar["Ux-Uy"] * sinTheta * (cosν ** 2 - sinv ** 2) \
                              + self.errvals['Ux'] * sinTheta * sinv * cosν - self.errvals['Uy'] * sinTheta * sinv * cosν
        self.covar["Uz-Sd"] = self.covar["Uz-Sd"] * cosTheta - self.covar["Ux-Sd"] * sinTheta * cosν \
                              - self.covar["Uy-Sd"] * sinv * sinTheta
        self.covar["Uxy-Uz"] = np.sqrt(self.covar["Ux-Uz"] ** 2 + self.covar["Uy-Uz"] ** 2)

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
                                  'PA': 'Pr',
                                  'H2O_density': 'pV',
                                  'RH_1_1_1': 'Rh',
                                  't_hmp': 'Ta',
                                  'e_hmp': 'Ea',
                                  'kh': 'volt_KH20',
                                  'q': 'Q'
                                  })

    # @njit
    def despike(self, arr, nstd: float = 4.5):
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
        cond_delta = (np.abs(np_spikey - np_fbewma) > delta)
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        return np_remove_outliers

    def despike_med_mod(self, df_column, win=800, fill_na=True, addNoise=False):
        """Detects and removes spikes using the scale and residuals of an RLM model of a moving window median filter;
        if residual > 3x modeled scale, then data are dropped and replaced with a random normal estimate plus linear
        interpolation.

        Args:
            df_column: datetime-indexed pandas Series of data with spikes
            win: size of moving window; default is 800 (40 seconds at 20 Hz).
            fill_na: fill gaps with linear interpolation with gaussian noise; defaults is true

        Returns:
            gap filled pandas Series

        """

        np_spikey = np.array(df_column)

        y = df_column.interpolate().bfill().ffill()
        x = df_column.rolling(window=win, center=True).median().bfill().ffill()

        X = sm.add_constant(x)
        mod_rlm = sm.RLM(y, X)
        mod_fit = mod_rlm.fit(maxiter=300, scale_est='mad')

        cond_delta = (np.abs(mod_fit.resid) > 3 * mod_fit.scale)
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        nanind = np.array(np.where(np.isnan(np_remove_outliers)))[0]

        data_out = pd.Series(np_remove_outliers, index=df_column.index)

        if fill_na:
            data_out = data_out.interpolate()
            data_outnaind = data_out.index[nanind]
            if addNoise:
                rando = np.random.normal(scale=mod_fit.scale, size=len(data_outnaind))
            else:
                rando = 0.0
            data_out.loc[data_outnaind] = data_out.loc[data_outnaind] + rando

        return data_out

    def despike_quart_filter(self, df_column, win=600, fill_na=True, top_quant=0.97, bot_quant=0.03, thresh=None):
        """Detects and removes spikes using a moving window quantile filter; if difference from median is > difference
        between 90% quartile and 10% quartile.

        Args:
            df_column: datetime-indexed pandas Series of data with spikes
            win: size of moving window; default is 1200 (1 minute at 20 Hz).
            fill_na: fill gaps with linear interpolation with gaussian noise; defaults is true
            top_quant: Upper Quantile that defines the inter-quartile range; default is 0.9
            bot_quant: Lower Quantile that defines the inter-quartile range; default is 0.1
            thresh: threshold of delta that defines a spike; default is None, which is the median of the rolling inter-quartile range

        Returns:
            gap filled pandas Series

        """
        np_spikey = np.array(df_column)

        # Get rolling statistics
        upper = pd.Series(df_column).rolling(window=win, center=True).quantile(top_quant).interpolate().bfill().ffill()
        lower = pd.Series(df_column).rolling(window=win, center=True).quantile(bot_quant).interpolate().bfill().ffill()
        med = pd.Series(df_column).rolling(window=win, center=True).median().interpolate().bfill().ffill()
        iqr = upper - lower

        if thresh:
            pass
        else:
            thresh = iqr

        cond_delta = (np.abs(np_spikey - med) > thresh)
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        nanind = np.array(np.where(np.isnan(np_remove_outliers)))[0]

        if fill_na:
            data_out = pd.Series(np_remove_outliers, index=df_column.index).interpolate()
            # data_outnaind = data_out.index[nanind]
            # rando = np.random.normal(scale=scl, size=len(data_outnaind))
            # data_out.loc[data_outnaind] = data_out.loc[data_outnaind] #+ rando
        else:
            data_out = pd.Series(np_remove_outliers, index=df_column.index)
        return data_out

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
    def calc_E(self, pV, T):
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
            >>> print(fluxcalc.calc_E(3.4,290.2))
            455362.68679999997

        """

        e = pV * T * self.Rv
        return e

    # @njit(parallel=True)
    def calc_Q(self, P, e):
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
            >>> print(fluxcalc.calc_Q(np.array([4003.6,4002.1]),np.array([717,710])))
            [0.11948162 0.11827882]

            >>> fluxcalc = CalcFlux()
            >>> print(fluxcalc.calc_Q(4003.6,717))
            0.11948162313727738

        """

        # molar mass of water vapor/ molar mass of dry air
        gamma = 0.622
        q = (gamma * e) / (P - 0.378 * e)
        return q

    def calc_tc_air_temp_sonic(self, Ts, pV, P):
        """Air temperature from sonic temperature, water vapor density, and atmospheric pressure from Campbell Scientific EasyFLux

        Args:
            Ts: Sonic Temperature (K)
            pV: h2O density (g m-3)
            P: Pressure (Pa)

        Returns:
            Ta Air Temperature

        References:
            Wallace and Hobbs 2006
        """

        pV = pV * 1000.0
        P_atm = 9.86923e-6 * P
        T_c1 = P_atm + (2 * self.Rv - 3.040446 * self.Rd) * pV * Ts
        T_c2 = P_atm * P_atm + (1.040446 * self.Rd * pV * Ts) * (
                1.040446 * self.Rd * pV * Ts) + 1.696000 * self.Rd * pV * P_atm * Ts
        T_c3 = 2 * pV * ((self.Rv - 1.944223 * self.Rd) + (self.Rv - self.Rd) * (
                self.Rv - 2.040446 * self.Rd) * pV * Ts / P_atm)

        return (T_c1 - np.sqrt(T_c2)) / T_c3

    # @njit
    def calc_Tsa(self, Ts, q):
        """Calculate air temperature from sonic temperature and specific humidity

        Args:
            Ts: Sonic Temperature (K)
            q: Specific Humidity (unitless)

        Returns:
            Tsa (air temperature derived from sonic temperature, K)

        References:
            Schotanus et al. (1983) doi:10.1007/BF00164332
            Also see Kaimal and Gaynor (1991) doi:10.1007/BF00119215
            Also See Van Dijk (2002)

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
            -83.69120608637277
        """
        return (-1 * (Ust ** 3) * Tsa) / (self.g * self.von_karman * Uz_Ta)

    # @numba.njit#(forceobj=True)
    def calc_Tsa_sonic_temp(self, Ts, P, pV):
        """
        Calculate the average sonic temperature
        :param Ts:
        :param P:
        :param pV:
        :param Rv:
        :return:
        """
        E = self.calc_E(pV, Ts)
        return -0.01645278052 * (
                -500 * P - 189 * E + np.sqrt(250000 * P ** 2 + 128220 * E * P + 35721 * E ** 2)) / pV / self.Rv

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
    def calc_Es(self, T):
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

    def lamb_func(self, x, varb):
        varib = dict(water=[2500800, -2360, 1.6, -0.06], ice=[2834100, -290, -4, 0])
        return varib[varb][0] + varib[varb][1] * x + varib[varb][2] * x ** 2 + varib[varb][3] * x ** 3

    def calc_latent_heat_of_vaporization(self, temperature, units='C'):
        """Calculates the latent heat of vaporization (Lambda) from temperature (deg C)

        Args:
            temperature: temperature
            units: units of temperature measurement; default is C; K also acceptable; F is for losers

        Returns:
            Specific Latent Heat of Condensation of Water (J/kg)
        References:
            From Rogers and Yau (1989) A Short Course in Cloud Physics
            https://en.wikipedia.org/wiki/Latent_heat
        """

        if units == 'K':
            temperature = self.convert_KtoC(temperature)
        else:
            pass

        return np.where(temperature >= 0, self.lamb_func(temperature,'water'),self.lamb_func(temperature,'ice')) # 'J/kg

    # @njit(parallel=True)
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
        df['Ts_K'] = self.convert_CtoK(df['Ts_ro'])
        df['E'] = self.calc_E(df['pV_ro'] * 0.001, df['Ts_K'])
        # convert air pressure from kPa to Pa
        df['Q'] = self.calc_Q(df['Pr_ro'] * 1000., df['E'])
        df['Tsa'] = self.calc_Tsa_air_temp_sonic(df['Ts_K'], df['Q'])
        df['Es'] = self.calc_Es(df['Tsa'])
        df['Sd'] = self.calc_Q(df['Pr_ro'], self.calc_Es(df['Tsa'])) - df['Q']
        return df

    # @numba.njit#(forceobj=True)
    def calc_pV(self, Ea, Ts):
        """Calculates water vapor density

        Args:
            Ea: Actual vapor pressure (Pa)
            Ts: Sonic Temperature (K)

        Returns:
            Water Vapor Density
        """
        return (Ea * 1000.0) / (self.Rv * Ts)

    def calc_max_covariance(self, x: np.ndarray, y: np.ndarray, lag: int = 10):
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
            else:
                # covariance for positive lags
                xy[i] = np.round(np.cov(x[i:], y[:-1 * i])[0][1], 8)
                # covariance for negative lags
                xy[-i] = np.round(np.cov(x[:-1 * i], x[i:])[0][1], 8)

        # convert dictionary to arrays
        keys = np.array(list(xy.keys()))
        vals = np.array(list(xy.values()))

        # get index and value for maximum positive covariance
        valmax, maxlagindex = self.findextreme(vals, ext='max')
        maxlag = keys[maxlagindex]
        maxcov = (maxlag, valmax)

        # get index and value for get maximum negative covariance
        valmin, minlagindex = self.findextreme(vals, ext='min')
        minlag = keys[minlagindex]
        mincov = (minlag, valmin)

        # get index and value for get maximum absolute covariance
        absmax, abslagindex = self.findextreme(vals, ext='min')
        absmaxlag = keys[abslagindex]
        abscov = (absmaxlag, absmax)

        return abscov, maxcov, mincov, xy

    def findextreme(self, vals, ext='abs'):
        """Used to find the extreme value and its index in an arrary.

        Args:
            vals: array
            ext: type of extreme; options are 'abs', 'min', and 'max'; default is 'abs'

        Returns:
            extreme value, index of extreme value
        """

        if ext == 'abs':
            vals = np.abs(vals)
            bigval = np.nanmax(vals)
        elif ext == 'max':
            bigval = np.nanmax(vals)
        elif ext == 'min':
            bigval = np.nanmin(vals)
        else:
            vals = np.abs(vals)
            bigval = np.nanmax(np.abs(vals))

        lagindex = np.where(vals == bigval)[0][0]

        return bigval, lagindex


    def calc_max_covariance_df(self, df: pd.DataFrame, colx: str, coly: str, lags: int = 10) -> list[float, int]:
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
    def coord_rotation(self, df: pd.DataFrame = None, Ux: str = 'Ux', Uy: str = 'Uy', Uz: str = 'Uz'):
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
        if df is None:
            df = self.df
        else:
            pass

        xmean = df[Ux].mean()
        ymean = df[Uy].mean()
        zmean = df[Uz].mean()
        Uxy = np.sqrt(xmean ** 2 + ymean ** 2)
        Uxyz = np.sqrt(xmean ** 2 + ymean ** 2 + zmean ** 2)
        self.cosv = xmean / Uxy
        self.sinv = ymean / Uxy
        self.sinTheta = zmean / Uxyz
        self.cosTheta = Uxy / Uxyz
        return self.cosv, self.sinv, self.sinTheta, self.cosTheta, Uxy, Uxyz

    def rotate_velocity_values(self, df: pd.DataFrame = None,
                               Ux: str = 'Ux', Uy: str = 'Uy', Uz: str = 'Uz') -> pd.DataFrame:
        """Rotate wind velocity values

        Args:
            df: Dataframe containing the wind velocity components
            Ux: Longitudinal component of the wind velocity (m s-1); aka u
            Uy: Lateral component of the wind velocity (m s-1); aka v
            Uz: Vertical component of the wind velocity (m s-1); aka w

        Returns:

        """
        if df is None:
            df = self.df
        else:
            pass

        if self.cosTheta is None:
            print("Please run coord_rotation")
            pass
        else:
            df['Uxr'] = df[Ux] * self.cosTheta * self.cosv + df[Uy] * self.cosTheta * self.sinv + df[Uz] * self.sinTheta
            df['Uyr'] = df[Uy] * self.cosv - df[Ux] * self.sinv
            df['Uzr'] = df[Uz] * self.cosTheta - df[Ux] * self.sinTheta * self.cosv - df[Uy] * self.sinTheta * self.sinv

            self.df = df
            return df


    def rotated_components_statistics(self, df: pd.DataFrame, Ux: str = 'Ux', Uy: str = 'Uy', Uz: str = 'Uz'):
        """Calculate the Average and Standard Deviations of the Rotated Velocity Components

        Args:
            df: Dataframe containing the wind velocity components
            Ux: Longitudinal component of the wind velocity (m s-1); aka u
            Uy: Lateral component of the wind velocity (m s-1); aka v
            Uz: Vertical component of the wind velocity (m s-1); aka w

        Returns:

        """
        if df is None:
            df = self.df
        else:
            pass

        self.avgvals['Uxr'] = df['Uxr'].mean()
        self.avgvals['Uyr'] = df['Uyr'].mean()
        self.avgvals['Uzr'] = df['Uzr'].mean()
        self.stdvals['Uxr'] = df['Uxr'].std()
        self.stdvals['Uyr'] = df['Uyr'].std()
        self.stdvals['Uzr'] = df['Uzr'].std()
        self.avgvals['Uav'] = self.avgvals['Ux'] * self.cosTheta * self.cosv + self.avgvals['Uy'] * self.cosTheta * self.sinv + self.avgvals['Uz'] * self.sinTheta
        return

    def dayfrac(self, df):
        return (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(1, unit='D')

    def calc_covar(self, Ux, Uy, Uz, Ts, Q, pV):
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

        self.covar['Ts-Ts'] = self.calc_cov(Ts, Ts)
        self.covar['Ux-Ux'] = self.calc_cov(Ux, Ux)
        self.covar['Uy-Uy'] = self.calc_cov(Uy, Uy)
        self.covar['Uz-Uy'] = self.calc_cov(Uz, Uz)
        self.covar['Q-Q'] = self.calc_cov(Q, Q)
        self.covar['pV-pV'] = self.calc_cov(pV, pV)
