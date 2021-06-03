# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import pandas as pd
import scipy
import numpy as np
import dask as dd
#Public Module EC

import numba

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

class CalcFluxWithKH20(object):
    """Determines H20 flux from input weather data, including a KH20 sensor, by the eddy covariance method.

    :param df: dataframe Weather Parameters for the Eddy Covariance Method; must be time-indexed and include Ux, Uy, Uz, Pr, Ea, and LnKH
    :return: Atmospheric Fluxes
    :notes:
    No High Pass Filtering or Trend Removal are Applied to the Data
    Time Series Data Are Moved Forward and Backward to Find Maximum Covariance Values
    Air Temperature and Sensible Heat Flux are Estimated From Sonic Temperature and Wind Data
    Other Corrections Include Transducer Shadowing, Traditional Coordinate Rotation, High Frequency Correctioons, and WPL"""

    def __init__(self, **kwargs):

        self.Rv = 461.51 # 'Water Vapor Gas Constant', 'J/[kg*K]'
        self.Ru = 8.314 # 'Universal Gas Constant', 'J/[kg*K]'
        self.Cpd = 1005 # 'Specific Heat of Dry Air', 'J/[kg*K]'
        self.Rd = 287.05 # 'Dry Air Gas Constant', 'J/[kg*K]'
        self.Co = 0.21  # Molar Fraction of Oxygen in the Atmosphere
        self.Mo = 0.032  # Molar Mass of Oxygen (gO2/mole)

        self.XKH20 = 1.412 # 'Path Length of KH20', 'cm'
        self.XKwC1 = -0.152214126 # First Order Coefficient in Vapor Density-KH20 Output Relationship, cm
        self.XKwC2 = -0.001667836 # Second Order Coefficient in Vapor Density-KH20 Output Relationship, cm
        self.directionKH20_U = 180
        self.UHeight = 3 # Height of Sonic Anemometer above Ground Surface', 'm'
        self.PathKH20_U = 0.1 # Separation Distance Between Sonic Anemometer and KH20', 'm', 0.1
        self.lag = 10 # number of lags to consider
        self.direction_bad_min = 0 # Clockwise Orientation from DirectionKH20_U
        self.direction_bad_max = 360 # Clockwise Orientation from DirectionKH20_U

        self.Kw = 1 # Extinction Coefficient of Water (m^3/[g*cm]) -instrument calibration
        self.Ko = -0.0045 # Extinction Coefficient of Oxygen (m^3/[g*cm]) -derived experimentally

        #Despiking Weather Parameters
        self.despikefields = ['Ux', 'Uy', 'Uz', 'Ts', 'volt_KH20', 'Pr', 'Ta', 'Rh']

        # Allow for update of input parameters
        # https://stackoverflow.com/questions/60418497/how-do-i-use-kwargs-in-python-3-class-init-function
        self.__dict__.update(kwargs)

        self.parameters = {
            'Ea': ['Actual Vapor Pressure', 'kPa'],
            'LnKH': ['Natural Log of Krypton Hygrometer Output', 'mV'],
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


    def runall(self, df):

        df = self.renamedf(df)

        if 'Ea' in df.columns:
            pass
        else:
            df['Ea'] = self.tetens(df['Ta'].to_numpy())

        if 'LnKH' in df.columns:
            pass
        else:
            df['LnKH'] = np.log(df['volt_KH20'].to_numpy())

        for col in self.despikefields:
            if col in df.columns:
                df[col] = self.despike(df[col].to_numpy(), nstd=4.5)


        df['Ts'] = self.convert_CtoK(df['Ts'].to_numpy())

        df['Ux'],df['Uy'],df['Uz'] = self.fix_csat(df['Ux'].to_numpy(),
                                                   df['Uy'].to_numpy(),
                                                   df['Uz'].to_numpy())

        # Calculate Sums and Means of Parameter Arrays
        df = self.calculated_parameters(df)

        # Calculate the Correct XKw Value for KH20
        XKw = self.XKwC1 + 2 * self.XKwC2 * (df['pV'].mean() * 1000.)
        self.Kw = XKw / self.XKH20



        # Calculate Covariances (Maximum Furthest From Zero With Sign in Lag Period)
        CovTs_Ts = df[['Ts', 'Ts']].cov().iloc[0,0] # location index needed because of same fields
        CovUx_Uy = df[['Ux', 'Uy']].cov().loc['Ux', 'Uy']  # CalcCovariance(IWP.Ux, IWP.Uy)
        CovUx_Uz = df[['Ux', 'Uz']].cov().loc['Ux', 'Uz']  # CalcCovariance(IWP.Ux, IWP.Uz)
        CovUy_Uz = df[['Uy', 'Uz']].cov().loc['Uy', 'Uz']  # CalcCovariance(IWP.Uy, IWP.Uz)

        CovTs_Q = self.calc_max_covariance(df, 'Ts', 'Q', self.lag)[0]
        CovUx_LnKH = self.calc_max_covariance(df, 'Ux', 'LnKH', self.lag)[0]
        CovUx_Q = self.calc_max_covariance(df, 'Ux', 'Q', self.lag)[0]
        CovUx_Sd = self.calc_max_covariance(df, 'Ux', 'Sd', self.lag)[0]
        CovUx_Ts = self.calc_max_covariance(df, 'Ux', 'Ts', self.lag)[0]
        CovUy_LnKH = self.calc_max_covariance(df, 'Uy', 'LnKH', self.lag)[0]
        CovUy_Q = self.calc_max_covariance(df, 'Uy', 'Q', self.lag)[0]
        CovUy_Sd = self.calc_max_covariance(df, 'Uy', 'Sd', self.lag)[0]
        CovUy_Ts = self.calc_max_covariance(df, 'Uy', 'Ts', self.lag)[0]
        CovUz_LnKH = self.calc_max_covariance(df, 'Uz', 'LnKH', self.lag)[0]
        CovUz_Q = self.calc_max_covariance(df, 'Uz', 'Q', self.lag)[0]
        CovUz_Sd = self.calc_max_covariance(df, 'Uz', 'Sd', self.lag)[0]
        CovUz_Ts = self.calc_max_covariance(df, 'Uz', 'Ts', self.lag)[0]

        # Traditional Coordinate Rotation
        cosν, sinν, sinTheta, cosTheta, Uxy, Uxyz = self.coord_rotation(df)

        # Find the Mean Squared Error of Velocity Components and Humidity
        UxMSE = self.calc_MSE(df['Ux'])
        UyMSE = self.calc_MSE(df['Uy'])
        UzMSE = self.calc_MSE(df['Uz'])
        QMSE = self.calc_MSE(df['Q'])

        # Correct Covariances for Coordinate Rotation
        Uz_Ts = CovUz_Ts * cosTheta - CovUx_Ts * sinTheta * cosν - CovUy_Ts * sinTheta * sinν
        if np.abs(Uz_Ts) >= np.abs(CovUz_Ts):
            CovUz_Ts = Uz_Ts

        Uz_LnKH = CovUz_LnKH * cosTheta - CovUx_LnKH * sinTheta * cosν - CovUy_LnKH * sinν * sinTheta
        if np.abs(Uz_LnKH) >= np.abs(CovUz_LnKH):
            CovUz_LnKH = Uz_LnKH
        CovUx_Q = CovUx_Q * cosTheta * cosν + CovUy_Q * cosTheta * sinν + CovUz_Q * sinTheta
        CovUy_Q = CovUy_Q * cosν - CovUx_Q * sinν
        CovUz_Q = CovUz_Q * cosTheta - CovUx_Q * sinTheta * cosν - CovUy_Q * sinν * sinTheta
        CovUx_Uz = CovUx_Uz * cosν * (cosTheta**2 - sinTheta**2) - 2 * CovUx_Uy * sinTheta * cosTheta * sinν * cosν + CovUy_Uz * sinν * (cosTheta**2 - sinTheta**2) - UxMSE * sinTheta * cosTheta * cosν**2 - UyMSE * sinTheta * cosTheta * sinν**2 + UzMSE * sinTheta * cosTheta
        CovUy_Uz = CovUy_Uz * cosTheta * cosν - CovUx_Uz * cosTheta * sinν - CovUx_Uy * sinTheta * (cosν**2 - sinν**2) + UxMSE * sinTheta * sinν * cosν - UyMSE * sinTheta * sinν * cosν
        CovUz_Sd = CovUz_Sd * cosTheta - CovUx_Sd * sinTheta * cosν - CovUy_Sd * sinν * sinTheta
        Uxy_Uz = np.sqrt(CovUx_Uz**2 + CovUy_Uz**2)
        Ustr = np.sqrt(Uxy_Uz)

        # Find Average Air Temperature From Average Sonic Temperature
        Tsa = self.calc_Tsa(df['Ts'].mean(), df['Pr'].mean(), df['pV'].mean())

        # Calculate the Latent Heat of Vaporization
        lamb = (2500800 - 2366.8 * (self.convert_KtoC(Tsa)))

        # Determine Vertical Wind and Water Vapor Density Covariance
        Uz_pV = (CovUz_LnKH / XKw) / 1000

        # Calculate the Correct Average Values of Some Key Parameters
        Cp = self.Cpd * (1 + 0.84 * df['Q'].mean())
        pD = (df['Pr'].mean() - df['E'].mean()) / (self.Rd * Tsa)
        p = pD + df['pV'].mean()

        # Calculate Variance of Air Temperature From Variance of Sonic Temperature
        StDevTa = np.sqrt(CovTs_Ts - 1.02 * df['Ts'].mean() * CovTs_Q - 0.2601 * QMSE * df['Ts'].mean()**2)
        Uz_Ta = CovUz_Ts - 0.07 * lamb * Uz_pV / (p * Cp)

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td(df['E'].mean())
        D = self.calc_Es(Tsa) - df['E'].mean()
        S = (self.calc_Q(df['Pr'].mean(), self.calc_Es(Tsa + 1)) - self.calc_Q(df['Pr'].mean(), self.calc_Es(Tsa - 1))) / 2

        # 'Determine Wind Direction
        WindDirection = np.arctan(df['Uy'].mean() / df['Ux'].mean()) * 180 / np.pi
        if df['Ux'].mean() < 0:
            WindDirection += 180 * np.sign(df['Uy'].mean())

        direction = self.directionKH20_U - WindDirection

        if direction < 0:
            direction += 360

        # 'Calculate the Lateral Separation Distance Projected Into the Mean Wind Direction
        pathlen = self.PathKH20_U * np.abs(np.sin((np.pi / 180) * direction))

        #'Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df['Uz'].std()
        UMean = df['Ux'].mean() * cosTheta * cosν + df['Uy'].mean() * cosTheta * sinν + df['Uz'].mean() * sinTheta

        #'Frequency Response Corrections (Massman, 2000 & 2001)
        tauB = (3600) / 2.8
        tauEKH20 = np.sqrt((0.01 / (4 * UMean)) **2 + (pathlen / (1.1 * UMean))**2)
        tauETs = np.sqrt((0.1 / (8.4 * UMean))**2)
        tauEMomentum = np.sqrt((0.1 / (5.7 * UMean))**2 + (0.1 / (2.8 * UMean))**2)

        #'Calculate ζ and Correct Values of Uᕽ and Uz_Ta
        L = self.calc_L(Ustr, Tsa, Uz_Ta)
        alpha, X = self.calc_AlphX(L)
        fX = X * UMean / self.UHeight
        B = 2 * np.pi * fX * tauB
        momentum = 2 * np.pi * fX * tauEMomentum
        _Ts = 2 * np.pi * fX * tauETs
        _KH20 = 2 * np.pi * fX * tauEKH20
        Ts = self.correct_spectral(B, alpha, _Ts)
        Uxy_Uz /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(Uxy_Uz)

        #'Recalculate L With New Uᕽ and Uz_Ta, and Calculate High Frequency Corrections
        L = self.calc_L(Ustr, Tsa, Uz_Ta / Ts)
        alpha, X = self.calc_AlphX(L)
        Ts = self.correct_spectral(B, alpha, _Ts)
        KH20 = self.correct_spectral(B, alpha, _KH20)

        #'Correct the Covariance Values
        Uz_Ta /= Ts
        Uz_pV /= KH20
        Uxy_Uz /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(Uxy_Uz)
        CovUz_Sd /= KH20
        exchange = ((p * Cp) / (S + Cp / lamb)) * CovUz_Sd

        #'KH20 Oxygen Correction
        Uz_pV += self.correct_KH20(Uz_Ta, df['Pr'].mean(), Tsa)

        #'Calculate New H and LE Values
        H = p * Cp * Uz_Ta
        lambdaE = lamb * Uz_pV

        #'Webb, Pearman and Leuning Correction
        lambdaE = lamb * p * Cp * Tsa * (1.0 + (1.0 / 0.622) * (df['pV'].mean() / pD)) * (Uz_pV + (df['pV'].mean() / Tsa) * Uz_Ta) / (p * Cp * Tsa + lamb * (1.0 + (1 / 0.622) * (df['pV'].mean() / pD)) * df['pV'].mean() * 0.07)

        #'Finish Output
        Tsa = self.convert_KtoC(Tsa)
        Td = self.convert_KtoC(Td)
        zeta = self.UHeight / L
        ET = lambdaE * self.get_Watts_to_H2O_conversion_factor(Tsa, (df.last_valid_index() - df.first_valid_index())/ pd.to_timedelta(1, unit='D'))
        #'Out.Parameters = CWP
        self.columns = ['Ta','Td','D', 'Ustr', 'zeta', 'H', 'StDevUz', 'StDevTa',  'direction', 'exchange', 'lambdaE', 'ET', 'Uxy']
        self.out = [Tsa, Td, D, Ustr, zeta, H, StDevUz, StDevTa,  direction, exchange,  lambdaE, ET, Uxy]
        return pd.Series(data=self.out,index=self.columns)

    def calc_LnKh(self, mvolts):
        return np.log(mvolts.to_numpy())

    def renamedf(self, df):
        return df.rename(columns={'T_SONIC':'Ts',
                                  'TA_1_1_1':'Ta',
                                  'amb_press':'Pr',
                                  'RH_1_1_1':'Rh',
                                  't_hmp':'Ta',
                                  'e_hmp':'Ea',
                                  'kh':'volt_KH20'
                                  })

    def despike(self, arr, nstd=4.5):
        """Removes spikes from parameter within a specified deviation from the mean.
        """
        stdd = np.nanstd(arr) * nstd
        avg = np.nanmean(arr)
        avgdiff = stdd - np.abs(arr - avg)
        y = np.where(avgdiff >= 0, arr, np.NaN)
        nans, x = np.isnan(y), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    def calc_Td(self, E):
        c0 = 207.98233
        c1 = -20.156028
        c2 = 0.46778925
        c3 = -0.0000092288067
        d0 = 1
        d1 = -0.13319669
        d2 = 0.0056577518
        d3 = -0.000075172865
        lne = np.log(E)
        return (c0 + c1 * lne + c2 * lne ** 2 + c3 * lne ** 3) / (d0 + d1 * lne + d2 * lne ** 2 + d3 * lne ** 3)

    def calc_Q(self, P, E):
        return (0.622 * E) / (P - 0.378 * E)

    def calc_E(self, pV, T):
        return pV * self.Rv * T

    def calc_L(self, Ust, Tsa, Uz_Ta):
        #removed negative sign
        return -1*(Ust ** 3) * Tsa / (9.8 * 0.4 * Uz_Ta)

    #@numba.njit#(forceobj=True)
    def calc_Tsa(self, Ts, P, pV, Rv=461.51):
        E = pV * self.Rv * Ts
        return -0.01645278052 * (
                    -500 * P - 189 * E + np.sqrt(250000 * P ** 2 + 128220 * E * P + 35721 * E ** 2)) / pV / Rv

    #@numba.jit(forceobj=True)
    def calc_AlphX(self, L):
        if (self.UHeight / L) <= 0:
            alph = 0.925
            X = 0.085
        else:
            alph = 1
            X = 2 - 1.915 / (1 + 0.5 * self.UHeight / L)
        return alph, X

    #@numba.jit(forceobj=True)
    def calc_Es(self,T):
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

    def calc_cov(self, p1, p2):
        # p1mean = np.mean(p1)
        # p2mean = np.mean(p2)
        sumproduct = 0
        for i in range(len(p1)):
            sumproduct += p1[i] * p2[i]

        return (sumproduct - (np.sum(p1) * np.sum(p2)) / len(p1)) / (len(p1) - 1)

   #@numba.njit#(forceobj=True)
    def calc_MSE(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def convert_KtoC(self, T):
        return T - 273.16

    def convert_CtoK(self, T):
        return T + 273.16

    def correct_KH20(self, Uz_Ta, P, T):
        """Calculates an additive correction for the KH20 due to cross sensitivity between H20 and 02 molecules.
        Uz_Ta = Covariance of Vertical Wind Component and Air Temperature (m*K/s)
        P = Air Pressure (Pa)
        T = Air Temperature (K)
        Kw = Extinction Coefficient of Water (m^3/[g*cm]) -instrument calibration
        Ko = Extinction Coefficient of Oxygen (m^3/[g*cm]) -derived experimentally
        returns KH20 Oxygen Correction
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
        d1 = -3.983035  # °C
        d2 = 301.797  # °C
        d3 = 522528.9  # °C2
        d4 = 69.34881  # °C
        d5 = 999.97495  # kg/m3
        return d5 * (1 - (temperature + d1) ** 2 * (temperature + d2) / (d3 * (temperature + d4)))  # 'kg/m^3

    def calc_latent_heat_of_vaporization(self, temperature):
        l0 = 2500800
        l1 = -2360
        l2 = 1.6
        l3 = -0.06
        return l0 + l1 * temperature + l2 * temperature ** 2 + l3 * temperature ** 3  # 'J/kg

    #@numba.njit#(forceobj=True)
    def fix_csat(self, Ux, Uy, Uz):

        CSAT3Inverse = [[-0.5, 0, 0.86602540378444],
                        [0.25, 0.4330127018922, 0.86602540378444],
                        [0.25, -0.4330127018922, 0.86602540378444]]
        CSAT3Transform = [[-1.3333333333333, 0.66666666666666, 0.66666666666666],
                          [0, 1.1547005383792, -1.1547005383792],
                          [0.3849001794597, 0.3849001794597, 0.3849001794597]]

        Ux_out = []
        Uy_out = []
        Uz_out = []

        for i in range(len(Ux)):
            u = {}
            u[0] = CSAT3Inverse[0][0] * Ux[i] + CSAT3Inverse[0][1] * Uy[i] + CSAT3Inverse[0][2] * Uz[i]
            u[1] = CSAT3Inverse[1][0] * Ux[i] + CSAT3Inverse[1][1] * Uy[i] + CSAT3Inverse[1][2] * Uz[i]
            u[2] = CSAT3Inverse[2][0] * Ux[i] + CSAT3Inverse[2][1] * Uy[i] + CSAT3Inverse[2][2] * Uz[i]

            scalar = (Ux[i] ** 2. + Uy[i] ** 2. + Uz[i] ** 2.) ** 0.5

            u[0] = u[0] / (0.68 + 0.32 * np.sin(np.arccos(u[0] / scalar)))
            u[1] = u[1] / (0.68 + 0.32 * np.sin(np.arccos(u[1] / scalar)))
            u[2] = u[2] / (0.68 + 0.32 * np.sin(np.arccos(u[2] / scalar)))

            Ux_out.append(CSAT3Transform[0][0] * u[0] + CSAT3Transform[0][1] * u[1] + CSAT3Transform[0][2] * u[2])
            Uy_out.append(CSAT3Transform[1][0] * u[0] + CSAT3Transform[1][1] * u[1] + CSAT3Transform[1][2] * u[2])
            Uz_out.append(CSAT3Transform[2][0] * u[0] + CSAT3Transform[2][1] * u[1] + CSAT3Transform[2][2] * u[2])

        return Ux_out, Uy_out, Uz_out

    # Calculated Weather Parameters
    # @numba.jit
    def calculated_parameters(self, df):
        df['pV'] = self.calc_pV(df['Ea'],df['Ts'])
        df['Tsa'] = self.calc_Tsa(df['Ts'], df['Pr'], df['pV'])
        df['E'] = self.calc_E(df['pV'], df['Tsa'])
        df['Q'] = self.calc_Q(df['Pr'], df['E'])
        df['Sd'] = self.calc_Q(df['Pr'], self.calc_Es(df['Tsa'])) - df['Q']
        return df

    #@numba.njit#(forceobj=True)
    def calc_pV(self, Ea, Ts):
        return (Ea * 1000.0) / (self.Rv * Ts)

    def calc_max_covariance(self, df, colx, coly, lags=10):
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

    #@numba.njit#(forceobj=True)
    def coord_rotation(self, df, Ux='Ux', Uy='Uy', Uz='Uz'):
        """Traditional Coordinate Rotation
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

    def dayfrac(self, df):
        return (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(1, unit='D')

    #@numba.njit#(forceobj=True)
    def tetens(self, t, a=0.611, b=17.502, c=240.97):
        """Tetens formula for computing the
        saturation vapor pressure of water from temperature; eq. 3.8

        t = temperature (C)
        a = constant (kPa)
        b = constant (dimensionless)
        c = constant (C)

        returns saturation vapor pressure ()
        """
        return a * np.exp((b * t) / (t + c))
