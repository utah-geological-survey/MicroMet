# Original code in Fortran by Larry Hipps (USU)
# Program to Calculate Fluxes from Complete Time Series Data
#  Revised 10 May 2022

import numpy as np
import pandas as pd

Ux = []
Uy = []
Uz = []
Ts = []
P = []
# vapor density
rhov = []

#  Define Values for Key Constants

# gas constant (J/mol*degK)
Runiv = 8.3144621
# gas constant of dry air (J/(kg*degK)
Rd = 287.05
# Gas constant of water vapor (J/(kg*degK))
Rv = 461.51
# specific heat for dry air at constant pressure at 27°C (J/(kg*degK))
cpd = 1005.0
# Molar Fraction of Oxygen in the Atmosphere
Co = 0.21
# Molar Mass of Oxygen (gO2/mole)
Mo = 0.032

#  Input the Separation Distance Between the CSAT and IRGA in m
dist = 0.0

#  Specify the Direction of the CSAT in Degrees and Height of Sensors in m
sonic_dir = 225
z = 3.52

d0 = 1.0
d1 = -1.3319669E-01
d2 = 5.6577518E-03
d3 = -7.5172865E-05


# @njit(parallel=True)
def shadow_correction(Ux, Uy, Uz):
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


def celcius_to_kelvin(Ts):
    return Ts + 273.16


def kpa_to_pa(P):
    return P * 1000.0


def get_sums(Ux, Uy, Uz, Ts, rhov, P):
    """Sum the Variables and Products Required for Later Calculations

    :param Ux: Longitudinal component of the wind velocity (m s-1); aka u
    :param Uy: Lateral component of the wind velocity (m s-1); aka v
    :param Uz: Vertical component of the wind velocity (m s-1); aka w
    :param Ts: Sonic Temperature (K)
    :param rhov:
    :param P: pressure (Pa)
    :return: sums of various constituents
    """
    sumx = np.sum(Ux)
    sumy = np.sum(Uy)
    sumz = np.sum(Uz)
    sumT = np.sum(Ts)
    sumv = np.sum(rhov)
    sump = np.sum(P)
    sumT2 = np.sum(np.square(Ts))
    sumuxux = np.sum(np.square(Ux))
    sumuxuy = np.sum(np.multiply(Ux, Uy))
    sumuxuz = np.sum(np.multiply(Ux, Uz))
    sumuyuy = np.sum(np.square(Uz))
    sumuyuz = np.sum(np.multiply(Uy, Uz))
    sumuzuz = np.sum(np.square(Uz))

    return sumx, sumy, sumz, sumT, sumv, sump, sumT2, sumuxux, sumuxuy, sumuxuz, sumuyuy, sumuyuz, sumuzuz


def ts_to_e(rhov, Ts, Rv=461.51):
    """Ideal Gas Law to calculate vapor pressure from water vapor density and temperature

    :param rhov: Density of water vapor in air (kg/m3)
    :param Ts: Sonic Temperature (K)
    :param Rv: Gas Constant of Water Vapor (J/(kg K))
    :return: e Actual Vapor Pressure (Pa)
    """
    e = rhov * Ts * Rv
    return e


def e_to_q(e, P):
    """Calculate Specific Humidity; Bolton 1980; (mass of water vapor)/ (mass of moist air)

    :param e: Actual Vapor Pressure (Pa)
    :param P: Air pressure (Pa)
    :return: Specific Humidity (unitless)
    """
    # molar mass of water vapor/ molar mass of dry air
    gamma = 0.622
    q = (gamma * e) / (P - 0.378 * e)
    return q


def sonic_to_air(Ts, q):
    """Convert sonic temperature into air temperature

    :param Ts: Sonic Temperature (K)
    :param q: Specific Humidity (unitless)
    :return: Tsa (air temperature from sonic temperature, K)
    """
    # Calculate air temperature from sonic temperature; Schotanus et al. (1983) doi:10.1007/BF00164332
    Tsa = Ts / (1 + 0.51 * q)
    return Tsa


# @numba.njit#(forceobj=True)
def calc_Tsa(Ts, P, pV, Rv=461.51):
    """
    Calculate the average sonic temperature
    :param Ts:
    :param P:
    :param pV:
    :param Rv:
    :return:
    """
    E = pV * Rv * Ts
    return -0.01645278052 * (
            -500 * P - 189 * E + np.sqrt(250000 * P ** 2 + 128220 * E * P + 35721 * E ** 2)) / pV / Rv


# @numba.njit#(forceobj=True)
def tetens(t, a=0.611, b=17.502, c=240.97):
    """Tetens formula for computing the
    saturation vapor pressure of water from temperature; eq. 3.8

    :param t: temperature (C)
    :param a: constant (kPa)
    :param b: constant (dimensionless)
    :param c: constant (C)
    :return: saturation vapor pressure (kPa)
    """
    return a * np.exp((b * t) / (t + c))


def calc_Es(T: float) -> float:
    """
    Saturation Vapor Pressure Equation
    :param T: Water temperature in Kelvin
    :return: Saturation Vapor Pressure (Pa)
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


def get_basic_sums(Ta, es, q):
    sumTa = np.sum(Ta)
    sumTa2 = np.sum(Ta ** 2)
    sume = np.sum(es)
    sumqm = np.sum(q)
    sumq2 = np.sum(q ** 2)
    return sumTa, sumTa2, sume, sumqm, sumq2


def get_basic_means(Ux, Uy, Uz, Ts, Ta, P, e, rhov, q):
    uxavg = np.mean(Ux)
    uyavg = np.mean(Uy)
    uzavg = np.mean(Uz)
    Tsavg = np.mean(Ts)
    Tavg = np.mean(Ta)
    Pavg = np.mean(P)
    eair = np.mean(e) / 1000.0
    rhovavg = np.mean(rhov)
    eavg = np.mean(e)
    qavg = np.mean(q)

    return uxavg, uyavg, uzavg, Tsavg, Tavg, Pavg, eair, rhovavg, eavg, qavg


def get_R_value(qavg, Rv=461.51, Rd=287.05):
    # gas constant of dry air (J/(kg*degK)
    # Gas constant of water vapor (J/(kg*degK))
    R = qavg * Rv + (1 - qavg) * Rd
    return R


def get_basic_variances(Ux, Uy, Uz, Ts, Ta):
    #  Calculate Averages and Standard Deviations for Velocities and Humidity
    stdux = np.std(Ux, ddof=1)
    stduy = np.std(Uy, ddof=1)
    stduz = np.std(Uz, ddof=1)
    stdTa = np.std(Ta, ddof=1)
    stdq = np.std(q, ddof=1)

    #  Uncorrected Variance of Sonic Temperature Which is Used Later
    varTs = np.var(Ts, ddof=1)

    #  Find Average of Square of Deviations from the Mean for Velocity Components and Humidity
    ux2bar = np.var(Ux)
    uy2bar = np.var(Uy)
    uz2bar = np.var(Uz)
    q2bar = np.var(q)
    return stdux, stduy, stduz, stdTa, stdq, varTs, ux2bar, uy2bar, uz2bar, q2bar


def calc_cp(qavg, cpd):
    #  Calculate the Correct Average Values for Some Key Variables
    cp = cpd * (1.0 + 0.84 * qavg)
    return cp


def calc_rho(Pavg, eavg, rhovavg):
    rhod = (Pavg - eavg) / (Rd * Ta)
    rho = rhod + rhovavg
    return rho


def calc_max_covariance_df(df: pd.DataFrame, colx: str, coly: str, lags: int = 10) -> [float, int]:
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
        df[f"{coly}_{i}"] = df[coly].shift[i]
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


# @njit
def calc_max_covariance_v4(x, y, lag: int = 10) -> [(int, float), (int, float), (int, float), dict]:
    """Shift Arrays in Both Directions and Calculate Covariances for Each Lag.
    This Will Account for Longitudinal Separation of Sensors or Any Synchronization Errors.

    :param x:
    :param y:
    :param lag:
    :return:

    """
    #  Shift the Wind and Scalar Arrays in Both Directions and Calculate Covariances for Each Lag.
    #  This Will Account for Longitudinal Separation of Sensors or Any Synchronization Errors.

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

    return maxcov, mincov, abscov, xy


def calc_cov(Ux, Uy, Uz):
    #  Calculate Covariances for Wind Components
    ux_ux = np.cov(Ux, Ux)[0][1]
    ux_uy = np.cov(Ux, Uy)[0][1]
    ux_uz = np.cov(Ux, Uz)[0][1]
    uy_uy = np.cov(Uy, Uy)[0][1]
    uy_uz = np.cov(Uy, Uz)[0][1]
    uz_uz = np.cov(Uz, Uz)[0][1]

    return ux_ux, ux_uy, ux_uz, uy_uy, uy_uz, uz_uz


def trad_coord_rotation(ux, uy, uz, u_Ts, u_rhov, u_q):
    """Traditional Coordinate Rotation

    :param uyavg:
    :param uxavg:
    :param uzavg:
    :return:
    """

    uxavg = np.mean(ux)
    uyavg = np.mean(uy)
    uzavg = np.mean(uz)

    eta = np.arctan(uyavg / uxavg)
    cosnu = (uxavg / np.sqrt(uxavg ** 2.0 + uyavg ** 2.0))
    sinnu = (uyavg / np.sqrt(uxavg ** 2.0 + uyavg ** 2.0))
    sintheta = (uzavg / np.sqrt(uxavg ** 2.0 + uyavg ** 2.0 + uzavg ** 2.0))
    costheta = (np.sqrt(uxavg ** 2.0 + uyavg ** 2.0) / np.sqrt(uxavg ** 2.0 + uyavg ** 2.0 + uzavg ** 2.0))

    #  Rotate the Velocity Values
    uxr = ux * costheta * cosnu + uy * costheta * sinnu + uz * sintheta
    uyr = uy * cosnu - ux * sinnu
    uzr = uz * costheta - ux * sintheta * cosnu - uy * sintheta * sinnu

    Uavg = uxavg * costheta * cosnu + uyavg * costheta * sinnu + uzavg * sintheta

    ux_Ts, uy_Ts, uz_Ts = u_Ts
    ux_rhov, uy_rhov, uz_rhov = u_rhov
    ux_q, uy_q, uz_q = u_q

    ux_uy = np.cov(ux, uy)[0][1]
    ux_uz = np.cov(ux, uz)[0][1]
    uy_uz = np.cov(uy, uz)[0][1]

    #  Correct Covariances for Coordinate Rotation
    uz_Tsr = uz_Ts * costheta - ux_Ts * sintheta * cosnu - uy_Ts * sintheta * sinnu

    if abs(uz_Tsr) >= abs(uz_Ts):
        uz_Ts = uz_Tsr

    uz_rhovr = uz_rhov * costheta - ux_rhov * sintheta * cosnu - uy_rhov * sinnu * sintheta

    if abs(uz_rhovr) >= abs(uz_rhov):
        uz_rhov = uz_rhovr

    ux_q = ux_q * costheta * cosnu + uy_q * costheta * sinnu + uz_q * sintheta
    uy_q = uy_q * cosnu - ux_q * sinnu
    uz_q = uz_q * costheta - ux_q * sintheta * cosnu - uy_q * sinnu * sintheta

    ux2bar = np.var(Ux)
    uy2bar = np.var(Uy)
    uz2bar = np.var(Uz)

    ux_uz = ux_uz * cosnu * (costheta ** 2 - sintheta ** 2) - 2.0 * ux_uy * sintheta * costheta * sinnu * cosnu + \
            uy_uz * sinnu * (costheta ** 2 - sintheta ** 2) - ux2bar * sintheta * costheta * cosnu ** 2 - \
            uy2bar * sintheta * costheta * sinnu ** 2 + uz2bar * sintheta * costheta

    uy_uz = uy_uz * costheta * cosnu - ux_uz * costheta * sinnu - ux_uy * sintheta * (cosnu ** 2 - sinnu ** 2) + \
            ux2bar * sintheta * sinnu * cosnu - uy2bar * sintheta * sinnu * cosnu

    uz_Un = np.sqrt(ux_uz ** 2.0 + uy_uz ** 2.0)
    ustar = np.sqrt(uz_Un)

    return uxr, uyr, uzr, Uavg, ustar


def latent_heat_vapor(Tavg):
    """Calculate Value of Latent Heat of Vaporization

    :param Tavg:
    :return:
    """
    return 2500800.0 - 2366.8 * (Tavg - 273.16)


def sat_vapor_press(Tavg, eavg, Pavg):
    """Determine Saturation Vapor Pressure of the Air; Uses Highly Accurate Wexler's Equations Modified by Hardy

    :param Tavg: Temperature (K)
    :param eavg: Actual Vapor Pressure (kPa)
    :param Pavg: Pressure (kPa
    :return:
    """

    #  Coefficients for Saturation Vapor Pressure Equation
    g0 = -2.8365744E03
    g1 = -6.028076559E03
    g2 = 1.954263612E01
    g3 = -2.737830188E-02
    g4 = 1.6261698E-05
    g5 = 7.0229056E-10
    g6 = -1.8680009E-13
    g7 = 2.7150305

    #  Coefficients for Dew Point Equation
    cc0 = 2.0798233E02
    cc1 = -2.0156028E01
    cc2 = 4.6778925E-01
    cc3 = -9.2288067E-06

    lnes = g0 * Tavg ** (
        -2) + g1 * Tavg ** 1.0 + g2 + g3 * Tavg + g4 * Tavg ** 2.0 + g5 * Tavg ** 3.0 + g6 * Tavg ** 4 + g7 * np.log(
        Tavg)
    es = np.exp(lnes)

    lne = np.log(eavg)
    Td = (cc0 + cc1 * lne + cc2 * lne ** 2.0 + cc3 * lne ** 3.0) / (d0 + d1 * lne + d2 * lne ** 2.0 + d3 * lne ** 3.0)
    D = es - eavg

    Tq1 = Ta - 1.0
    lnes = g0 * Tq1 ** (-2) + g1 * Tq1 ** (
        -1) + g2 + g3 * Tq1 + g4 * Tq1 ** 2.0 + g5 * Tq1 ** 3.0 + g6 * Tq1 ** 4 + g7 * np.log(Tq1)
    es = np.exp(lnes)

    qs1 = (0.622 * es) / (Pavg - 0.378 * es)
    Tq2 = Ta + 1.0

    lnes = g0 * Tq2 ** (-2) + g1 * Tq2 ** (
        -1) + g2 + g3 * Tq2 + g4 * Tq2 ** 2.0 + g5 * Tq2 ** 3.0 + g6 * Tq2 ** 4 + g7 * np.log(Tq2)
    es = np.exp(lnes)

    qs2 = (0.622 * es) / (Pavg - 0.378 * es)
    s = (qs2 - qs1) / 2.0
    return es, s


def get_wind_dir(Ux, Uy, sonic_dir):
    """Determine Wind Direction

    :param Ux:
    :param Uy:
    :param sonic_dir:
    :return:
    """

    uxavg = np.mean(Ux)
    uyavg = np.mean(Uy)

    V = np.sqrt(uxavg ** 2 + uyavg ** 2)
    wind_dir = np.arctan(uyavg / uxavg) * 180.0 / np.pi
    if uxavg < 0:
        if uyavg >= 0:
            wind_dir = wind_dir + 180.0
        else:
            wind_dir = wind_dir - 180.0

    wind_compass = -1.0 * wind_dir + sonic_dir

    if wind_compass < 0:
        wind_compass = wind_compass + 360
    elif wind_compass > 360:
        wind_compass = wind_compass - 360

    phi = (np.pi / 180.0) * wind_compass

    return wind_compass, phi, V


def freq_response_massman(Uavg, Tavg, ustar, uz_Ta, uz_Un, uz_rhov, lv):
    #  Frequency Response Corrections for Path Length and Frequency Response (Massman 2000 & 2001)
    taob = (60 * 60) / 2.8
    taoeKH20 = np.sqrt(((1.0 / 100) / (4.0 * Uavg)) ** 2 + (path / (1.1 * Uavg)) ** 2.0)
    taoeTs = np.sqrt(((10.0 / 100) / (8.4 * Uavg)) ** 2)
    taoeMomentum = np.sqrt(((10.0 / 100) / (5.7 * Uavg)) ** 2 + ((10.0 / 100) / (2.8 * Uavg)) ** 2)

    #  Calculate z/L and Correct Values of Ustar and Covariance Vertical Wind and Air Temperature -- uz_Ta
    L = -1 * (ustar ** 3) * Tavg / (9.8 * 0.4 * uz_Ta)
    if z / L <= 0.0:
        alfa = 0.925
        nx = 0.085
    else:
        alfa = 1.0
        nx = 2.0 - 1.915 / (1.0 + 0.5 * z / L)

    fx = nx * Uavg / z
    b = 2.0 * np.pi * fx * taob
    pMomentum = 2.0 * np.pi * fx * taoeMomentum
    pTs = 2.0 * np.pi * fx * taoeTs
    pkh20 = 2.0 * np.pi * fx * taoeKH20

    rMomentum = ((b ** alfa) / ((b ** alfa) + 1.0)) * ((b ** alfa) / (b ** alfa + pMomentum ** alfa)) * (
                1.0 / ((pMomentum ** alfa) + 1.0))

    rTs = ((b ** alfa) / ((b ** alfa) + 1.0)) * ((b ** alfa) / (b ** alfa + pTs ** alfa)) * (
                1.0 / ((pTs ** alfa) + 1.0))

    uz_Un = uz_Un / rMomentum
    ustarn = np.sqrt(uz_Un)
    uz_Tan = uz_Ta / rTs

    #  Re-calculate Monin-Obukov Length with New Ustar and Uz_Ta.
    #  Calculate High Frequency Corrections
    L = -(ustarn ** 3) * Tavg / (9.8 * 0.4 * uz_Tan)
    if z / L <= 0.0:
        alfa = 0.925
        nx = 0.085
    else:
        alfa = 1.0
        nx = 2.0 - 1.915 / (1.0 + 0.5 * z / L)

    rMomentum = ((b ** alfa) / ((b ** alfa) + 1.0)) * ((b ** alfa) / (b ** alfa + pMomentum ** alfa)) * (
                1.0 / ((pMomentum ** alfa) + 1.0))

    rTs = ((b ** alfa) / ((b ** alfa) + 1.0)) * ((b ** alfa) / (b ** alfa + pTs ** alfa)) * (
                1.0 / ((pTs ** alfa) + 1.0))

    rKH20 = ((b ** alfa) / ((b ** alfa) + 1.0)) * ((b ** alfa) / (b ** alfa + pkh20 ** alfa)) * (
                1.0 / ((pkh20 ** alfa) + 1.0))

    #  Correct the Covariance Values for High Frequency Effects

    uz_Ta = uz_Ta / rTs
    uz_rhov = uz_rhov / rKH20
    uz_Un = uz_Un / rMomentum
    ustar = np.sqrt(uz_Un)
    zeta = z / L
    gamma = cp / lv

    return rMomentum, rTs, rKH20, L, uz_Ta, uz_rhov, uz_Un, ustar, zeta, gamma


def H_LE(rho, cp, uz_Ta, Lv, uz_rhov):
    #
    #  Calculate New H and LE Values
    #

    H = rho * cp * uz_Ta
    LE = Lv * uz_rhov
    return H, LE


def webb_pearman_leuning(Lv, rho, cp, Tavg, rhovavg, rhod):
    #
    #  Webb, Pearman and Leuning Correction
    #

    LE = Lv * rho * cp * Tavg * (1.0 + (1.0 / 0.622) *
                                 (rhovavg / rhod)) * \
         (uz_rhov + (rhovavg / Tavg) * uz_Ta) / (
                     rho * cp * Tavg + Lv * (1.0 + (1 / 0.622) * (rhovavg / rhod)) * rhovavg * 0.07)
    return LE


e = ts_to_e(rhov, Ts)
q = e_to_q(e, P)
Ta = sonic_to_air(Ts, q)
es = calc_Es(Ta)
qs = (0.622 * es) / (P - 0.378 * es)

Tsavg = np.mean(Ts)
Tavg = np.mean(Ta)
qavg = np.mean(q)
#  Uncorrected Variance of Sonic Temperature
varTs = np.var(Ts, ddof=1)

if q_Ts:
    pass
else:
    q_Ts = np.cov(q, Ts)[0][1]

q2bar = np.var(q)
#  Calculate Variance of Air Temperature From Variance of Sonic Temperature
varTa = varTs - 1.02 * Tsavg * q_Ts - (0.51 ** 2.0) * q2bar * Tsavg ** 2.0
stdTa1 = np.sqrt(varTa)
cp = cpd * (1.0 + 0.84 * qavg)
Lv = 2500800.0 - 2366.8 * (Tavg - 273.16)

ux_Ts, uy_Ts, ux_rhov, uy_rhov, uz_rhov, uzTs_shift, uzrhov_shift, zT, zrhov = cmax_covariance_v2(10, Ux, Uy, Uz, Ts,
                                                                                                  rhov, q)

# Calculate max variance to close separation between sensors
velocities = {"Ux": Ux, "Uy": Uy, "Uz": Uz}
covariance_variables = {"Ts": Ts, "rhov": rhov, "q": q}

max_covariances = {}

for ik, iv in velocities.items():
    for jk, jv in covariance_variables.items():
        max_covariances[f"{ik}-{jk}"] = calc_max_covariance_v4(iv, jv)[2]

ux_Ts = max_covariances["Ux-Ts"][1]
uy_Ts = max_covariances["Uy-Ts"][1]
uz_Ts = max_covariances["Uz-Ts"][1]

ux_rhov = max_covariances["Ux-rhov"][1]
uy_rhov = max_covariances["Uy-rhov"][1]
uz_rhov = max_covariances["Uz-rhov"][1]

uz_Ta = uz_Ts - 0.06 * Lv * uz_rhov / (rhov * cp)

wind_compass, phi = get_wind_dir(Ux, Uy, sonic_dir)

#  Calculate the Lateral Separation Distance Projected into the Mean Wind Direction.
#  Only Needed When IRGA and Sonic Are Separated.
path = dist * np.abs(np.sin(phi))

#
#  Calculate Variance, Skewness, and Kurtosis of instantaneous rhov flux
#

sumrhov1 = np.sum(rhov)
sumuz1 = np.sum(Uz)

avgrhov = np.mean(rhov)
avguz = np.mean(Uz)

#
#  Change the Various T Values to C
#  Mean of All Individual Tair from Tsonic Values = Tavg
#  Average Air Temperature from Average Sonic Temperature = Tair_avg
#

Tavg = Tavg - 273.16
Tsair = Tsavg / (1 + 0.51 * qavg)
Tsair = Tsair - 273.16
Td = Td - 273.16

#
#  Sum Individual ET Values for Daily Calculation
#

if LE > 0.0:
    ET = ET + (LE * 60 * 60) / Lv

# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import pandas as pd
import scipy
import numpy as np
import dask as dd
# Public Module EC

import numba
from numba import njit


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
    Other Corrections Include Transducer Shadowing, Traditional Coordinate Rotation, High Frequency Corrections, and WPL"""

    def __init__(self, **kwargs):

        self.Rv = 461.51  # 'Water Vapor Gas Constant', 'J/[kg*K]'
        self.Ru = 8.314  # 'Universal Gas Constant', 'J/[kg*K]'
        self.Cpd = 1005.0  # 'Specific Heat of Dry Air', 'J/[kg*K]'
        self.Rd = 287.05  # 'Dry Air Gas Constant', 'J/[kg*K]'
        self.Co = 0.21  # Molar Fraction of Oxygen in the Atmosphere
        self.Mo = 0.032  # Molar Mass of Oxygen (gO2/mole)

        self.XKH20 = 1.412  # 'Path Length of KH20', 'cm'
        self.XKwC1 = -0.152214126  # First Order Coefficient in Vapor Density-KH20 Output Relationship, cm
        self.XKwC2 = -0.001667836  # Second Order Coefficient in Vapor Density-KH20 Output Relationship, cm
        self.directionKH20_U = 180
        self.UHeight = 3  # Height of Sonic Anemometer above Ground Surface', 'm'
        self.PathKH20_U = 0.1  # Separation Distance Between Sonic Anemometer and KH20', 'm', 0.1
        self.lag = 10  # number of lags to consider
        self.direction_bad_min = 0  # Clockwise Orientation from DirectionKH20_U
        self.direction_bad_max = 360  # Clockwise Orientation from DirectionKH20_U

        self.Kw = 1  # Extinction Coefficient of Water (m^3/[g*cm]) -instrument calibration
        self.Ko = -0.0045  # Extinction Coefficient of Oxygen (m^3/[g*cm]) -derived experimentally

        # Despiking Weather Parameters
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

        df.loc[:, 'Ts'] = self.convert_CtoK(df['Ts'].to_numpy())
        df.loc[:, 'Ta'] = self.convert_CtoK(df['Ta'].to_numpy())

        df['Ux'], df['Uy'], df['Uz'] = self.fix_csat(df['Ux'].to_numpy(),
                                                     df['Uy'].to_numpy(),
                                                     df['Uz'].to_numpy())

        # Calculate Sums and Means of Parameter Arrays
        df = self.calculated_parameters(df)

        # Calculate the Correct XKw Value for KH20
        XKw = self.XKwC1 + 2 * self.XKwC2 * (df['pV'].mean() * 1000.)
        self.Kw = XKw / self.XKH20

        # Calculate Covariances (Maximum Furthest From Zero With Sign in Lag Period)
        CovTs_Ts = df[['Ts', 'Ts']].cov().iloc[0, 0]  # location index needed because of same fields
        CovUx_Uy = df[['Ux', 'Uy']].cov().loc['Ux', 'Uy']  # CalcCovariance(IWP.Ux, IWP.Uy)
        CovUx_Uz = df[['Ux', 'Uz']].cov().loc['Ux', 'Uz']  # CalcCovariance(IWP.Ux, IWP.Uz)
        CovUy_Uz = df[['Uy', 'Uz']].cov().loc['Uy', 'Uz']  # CalcCovariance(IWP.Uy, IWP.Uz)

        # Calculate max variance to close separation between sensors
        velocities = {"Ux": df['Ux'].to_numpy(), "Uy": df['Uy'].to_numpy(), "Uz": df['Uz'].to_numpy()}
        covariance_variables = {"Ts": df['Ts'].to_numpy(), "rhov": rhov, "Q": df['Q'].to_numpy()}

        self.Cov = {}

        for ik, iv in velocities.items():
            for jk, jv in covariance_variables.items():
                self.Cov[f"{ik}_{jk}"] = self.calc_max_covariance(iv, jv)[2]

        # Traditional Coordinate Rotation
        cosν, sinν, sinTheta, cosTheta, Uxy, Uxyz = self.coord_rotation(df)

        # Find the Mean Squared Error of Velocity Components and Humidity
        UxMSE = self.calc_MSE(df['Ux'])
        UyMSE = self.calc_MSE(df['Uy'])
        UzMSE = self.calc_MSE(df['Uz'])
        QMSE = self.calc_MSE(df['Q'])

        # Correct Covariances for Coordinate Rotation
        Uz_Ts = self.Cov["Uz_Ts"] * cosTheta - self.Cov["Ux_Ts"] * sinTheta * cosν - self.Cov["Uy_Ts"] * sinTheta * sinν
        if np.abs(Uz_Ts) >= np.abs(Cov["Uz_Ts"]):
            CovUz_Ts = Uz_Ts

        Uz_LnKH = self.Cov["Uz_LnKH"] * cosTheta - self.Cov["Ux_LnKH"] * sinTheta * cosν - self.Cov[
            "Uy_LnKH"] * sinν * sinTheta
        if np.abs(Uz_LnKH) >= np.abs(CovUz_LnKH):
            CovUz_LnKH = Uz_LnKH
        CovUx_Q = CovUx_Q * cosTheta * cosν + CovUy_Q * cosTheta * sinν + CovUz_Q * sinTheta
        CovUy_Q = CovUy_Q * cosν - CovUx_Q * sinν
        CovUz_Q = CovUz_Q * cosTheta - CovUx_Q * sinTheta * cosν - CovUy_Q * sinν * sinTheta
        CovUx_Uz = CovUx_Uz * cosν * (
                    cosTheta ** 2 - sinTheta ** 2) - 2 * CovUx_Uy * sinTheta * cosTheta * sinν * cosν + CovUy_Uz * sinν * (
                               cosTheta ** 2 - sinTheta ** 2) - UxMSE * sinTheta * cosTheta * cosν ** 2 - UyMSE * sinTheta * cosTheta * sinν ** 2 + UzMSE * sinTheta * cosTheta
        CovUy_Uz = CovUy_Uz * cosTheta * cosν - CovUx_Uz * cosTheta * sinν - CovUx_Uy * sinTheta * (
                    cosν ** 2 - sinν ** 2) + UxMSE * sinTheta * sinν * cosν - UyMSE * sinTheta * sinν * cosν
        CovUz_Sd = CovUz_Sd * cosTheta - CovUx_Sd * sinTheta * cosν - CovUy_Sd * sinν * sinTheta
        Uxy_Uz = np.sqrt(CovUx_Uz ** 2 + CovUy_Uz ** 2)
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
        StDevTa = np.sqrt(CovTs_Ts - 1.02 * df['Ts'].mean() * CovTs_Q - 0.2601 * QMSE * df['Ts'].mean() ** 2)
        Uz_Ta = CovUz_Ts - 0.07 * lamb * Uz_pV / (p * Cp)

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td(df['E'].mean())
        D = self.calc_Es(Tsa) - df['E'].mean()
        S = (self.calc_Q(df['Pr'].mean(), self.calc_Es(Tsa + 1)) - self.calc_Q(df['Pr'].mean(),
                                                                               self.calc_Es(Tsa - 1))) / 2

        # Determine Wind Direction
        WindDirection = np.arctan(df['Uy'].mean() / df['Ux'].mean()) * 180 / np.pi
        if df['Ux'].mean() < 0:
            WindDirection += 180 * np.sign(df['Uy'].mean())

        direction = self.directionKH20_U - WindDirection

        if direction < 0:
            direction += 360

        # Calculate the Lateral Separation Distance Projected Into the Mean Wind Direction
        pathlen = self.PathKH20_U * np.abs(np.sin((np.pi / 180) * direction))

        # Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df['Uz'].std()
        UMean = df['Ux'].mean() * cosTheta * cosν + df['Uy'].mean() * cosTheta * sinν + df['Uz'].mean() * sinTheta

        # Frequency Response Corrections (Massman, 2000 & 2001)
        tauB = (3600) / 2.8
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
        Uxy_Uz /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(Uxy_Uz)

        # Recalculate L With New Uᕽ and Uz_Ta, and Calculate High Frequency Corrections
        L = self.calc_L(Ustr, Tsa, Uz_Ta / Ts)
        alpha, X = self.calc_AlphX(L)
        Ts = self.correct_spectral(B, alpha, _Ts)
        KH20 = self.correct_spectral(B, alpha, _KH20)

        # Correct the Covariance Values
        Uz_Ta /= Ts
        Uz_pV /= KH20
        Uxy_Uz /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(Uxy_Uz)
        CovUz_Sd /= KH20
        exchange = ((p * Cp) / (S + Cp / lamb)) * CovUz_Sd

        # KH20 Oxygen Correction
        Uz_pV += self.correct_KH20(Uz_Ta, df['Pr'].mean(), Tsa)

        # Calculate New H and LE Values
        H = p * Cp * Uz_Ta
        lambdaE = lamb * Uz_pV

        # Webb, Pearman and Leuning Correction
        lambdaE = lamb * p * Cp * Tsa * (1.0 + (1.0 / 0.622) * (df['pV'].mean() / pD)) * (
                    Uz_pV + (df['pV'].mean() / Tsa) * Uz_Ta) / (
                              p * Cp * Tsa + lamb * (1.0 + (1 / 0.622) * (df['pV'].mean() / pD)) * df[
                          'pV'].mean() * 0.07)

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

    def calc_LnKh(self, mvolts):
        return np.log(mvolts.to_numpy())

    def renamedf(self, df):
        return df.rename(columns={'T_SONIC': 'Ts',
                                  'TA_1_1_1': 'Ta',
                                  'amb_press': 'Pr',
                                  'RH_1_1_1': 'Rh',
                                  't_hmp': 'Ta',
                                  'e_hmp': 'Ea',
                                  'kh': 'volt_KH20'
                                  })

    def despike(self, arr, nstd: float = 4.5):
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
        """
        Dew point equation
        :param E: Water vapour pressure at saturation
        :return: Td dew point
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
        return (c0 + c1 * lne + c2 * lne ** 2 + c3 * lne ** 3) / (d0 + d1 * lne + d2 * lne ** 2 + d3 * lne ** 3)

    def calc_Q(self, P, E):
        return (0.622 * E) / (P - 0.378 * E)

    def calc_E(self, pV, T):
        return pV * self.Rv * T

    @njit(parallel=True)
    def calc_L(self, Ust, Tsa, Uz_Ta):
        # removed negative sign
        return -1 * (Ust ** 3) * Tsa / (9.8 * 0.4 * Uz_Ta)

    @njit(parallel=True)
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

    @njit(parallel=True)
    def calc_AlphX(self, L):
        if (self.UHeight / L) <= 0:
            alph = 0.925
            X = 0.085
        else:
            alph = 1
            X = 2 - 1.915 / (1 + 0.5 * self.UHeight / L)
        return alph, X

    @njit(parallel=True)
    def tetens(self, t, a=0.611, b=17.502, c=240.97):
        """Tetens formula for computing the
        saturation vapor pressure of water from temperature; eq. 3.8

        :param t: temperature (C)
        :param a: constant (kPa)
        :param b: constant (dimensionless)
        :param c: constant (C)
        :return: saturation vapor pressure (kPa)
        """
        return a * np.exp((b * t) / (t + c))

    @njit(parallel=True)
    def calc_Es(self, T: float) -> float:
        """
        Saturation Vapor Pressure Equation
        :param T: Water temperature in Kelvin
        :return: Saturation Vapor Pressure (Pa)
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

    @njit(parallel=True)
    def calc_cov(self, p1, p2):
        """
        Calculate covariance between two variables
        :param p1:
        :param p2:
        :return:
        """
        sumproduct = np.sum(p1 * p2)
        return (sumproduct - (np.sum(p1) * np.sum(p2)) / len(p1)) / (len(p1) - 1)

    @njit(parallel=True)
    def calc_MSE(self, y):
        """
        Calculate mean standard error
        :param y:
        :return:
        """
        return np.mean((y - np.mean(y)) ** 2)

    def convert_KtoC(self, T):
        """
        Convert Kelvin to Celcius
        :param T: Temperature in Kelvin
        :return: Temperature in Celcius
        """
        return T - 273.16

    def convert_CtoK(self, T):
        """
        Convert Celcius to Kelvin
        :param T: Temperature in Celcius degrees
        :return: Temperature in Kelvin
        """
        return T + 273.16

    @njit(parallel=True)
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

    @njit(parallel=True)
    def correct_spectral(self, B, alpha, varib):
        B_alpha = B ** alpha
        V_alpha = varib ** alpha
        return (B_alpha / (B_alpha + 1)) * (B_alpha / (B_alpha + V_alpha)) * (1 / (V_alpha + 1))

    def get_Watts_to_H2O_conversion_factor(self, temperature, day_fraction):
        to_inches = 25.4
        return (self.calc_water_density(temperature) * 86.4 * day_fraction) / (
                self.calc_latent_heat_of_vaporization(temperature) * to_inches)

    @njit(parallel=True)
    def calc_water_density(self, temperature):
        """Calculate the density of water (kg/m3) for a given temperature (C)"""
        d1 = -3.983035  # °C
        d2 = 301.797  # °C
        d3 = 522528.9  # °C2
        d4 = 69.34881  # °C
        d5 = 999.97495  # kg/m3
        return d5 * (1 - (temperature + d1) ** 2 * (temperature + d2) / (d3 * (temperature + d4)))  # 'kg/m^3

    @njit(parallel=True)
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

            # Adjustment Factors for Each Component
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

    def calculated_parameters(self, df):
        """Calculated Weather Parameters

        :param df:
        :return:
        """
        df['pV'] = self.calc_pV(df['Ea'], df['Ts'])
        df['Tsa'] = self.calc_Tsa(df['Ts'], df['Pr'], df['pV'])
        df['E'] = self.calc_E(df['pV'], df['Tsa'])
        df['Q'] = self.calc_Q(df['Pr'], df['E'])
        df['Sd'] = self.calc_Q(df['Pr'], self.calc_Es(df['Tsa'])) - df['Q']
        return df

    @njit(parallel=True)
    def calc_pV(self, Ea, Ts):
        return (Ea * 1000.0) / (self.Rv * Ts)

    @njit
    def calc_max_covariance(self, x, y, lag: int = 10) -> [(int, float), (int, float), (int, float), dict]:
        """Shift Arrays in Both Directions and Calculate Covariances for Each Lag.
        This Will Account for Longitudinal Separation of Sensors or Any Synchronization Errors.

        :param x:
        :param y:
        :param lag:
        :return:

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

        return maxcov, mincov, abscov, xy

    # @numba.njit#(forceobj=True)
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

    def coordinate_rotation(self, Ux, Uy, Uz):
        """Traditional Coordinate Rotation

        :param Ux: Longitudinal component of the wind velocity (m s-1); aka u
        :param Uy: Lateral component of the wind velocity (m s-1); aka v
        :param Uz: Vertical component of the wind velocity (m s-1); aka w
        :return:
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
