# -*- coding: utf-8 -*-
"""
Library of functions for meteorology.
modified from: http://python.hydrology-amsterdam.nl/

Meteorological function names
=============================

    - cp_calc:    Calculate specific heat
    - Delta_calc: Calculate slope of vapour pressure curve
    - es_calc:    Calculate saturation vapour pressures
    - ea_calc:    Calculate actual vapour pressures
    - gamma_calc: Calculate psychrometric constant
    - L_calc:     Calculate latent heat of vapourisation
    - pottemp:    Calculate potential temperature (1000 hPa reference pressure)
    - rho_calc:   Calculate air density
    - sun_NR:     Maximum sunshine duration [h] and extraterrestrial radiation [J/day]
    - vpd_calc:   Calculate vapour pressure deficits
    - windvec:    Calculate average wind direction and speed

Module requires and imports math and scipy modules.

Tested for compatibility with Python 2.7.

Function descriptions
=====================

Functions for calculation of potential and actual evaporation
from meteorological data.

Potential and actual evaporation functions
==========================================

        - E0: Calculate Penman (1948, 1956) open water evaporation.
        - Em: Calculate evaporation according to Makkink (1965).
        - Ept: Calculate evaporation according to Priestley and Taylor (1972).
        - ET0pm: Calculate Penman Monteith reference evaporation short grass.
        - Epm: Calculate Penman-Monteith evaporation (actual).
        - ra: Calculate aerodynamic resistance from windspeed and
          roughnes parameters.
        - tvardry: calculate sensible heat flux from temperature variations.
        - gash79: Gash (1979) analytical rainfall interception model.

Requires and imports scipy and meteolib modules.
Compatible with Python 2.7.3.

Function descriptions
=====================

"""

# Load relevant python functions

import math  # import math library
import numpy as np
import scipy  # import scientific python functions

__author__ = "Maarten J. Waterloo <m.j.waterloo@vu.nl> and J. Delsman"
__version__ = "1.0"
__date__ = "November 2014"


def meteolib():
    """
    A library of functions for the calculation of micrometeorological parameters.
    This prints a list of functions, and information about the author, version, and last modification date.

    Functions
    ---------
    - cp_calc: Calculate specific heat.
    - Delta_calc: Calculate slope of the vapour pressure curve.
    - ea_calc: Calculate actual vapour pressures.
    - es_calc: Calculate saturation vapour pressures.
    - gamma_calc: Calculate psychrometric constant.
    - L_calc: Calculate latent heat of vapourisation.
    - pottemp: Calculate potential temperature (1000 hPa reference pressure).
    - rho_calc: Calculate air density.
    - sun_NR: Calculate extraterrestrial radiation and daylength.
    - vpd_calc: Calculate vapour pressure deficits.
    - windvec: Calculate average wind direction and speed.

    Author: {}
    Version: {}
    Date: {}
    """.format(__author__, __version__, __date__)
    print(meteolib.__doc__)




def convert_to_array(*args):
    """
    Function to convert input parameters in as lists or tuples to
    arrays, while leaving single values intact.
    Test function for single values or valid array parameter input.
    Parameters:
        args (array, list, tuple, int, float): Input values for functions.
    Returns:
        valid_args (array, int, float): Valid single value or array function input.
    Examples
    --------
        >>> convert_to_array(12.76)
        12.76
        >>> convert_to_array([(1,2,3,4,5),(6,7,8,9)])
        array([(1, 2, 3, 4, 5), (6, 7, 8, 9)], dtype=object)
        >>> x=[1.2,3.6,0.8,1.7]
        >>> convert_to_array(x)
        array([ 1.2,  3.6,  0.8,  1.7])
        >>> convert_to_array('This is a string')
        'This is a string'
    """
    valid_args = []
    for a in args:
        if isinstance(a, (list, tuple)):
            valid_args.append(np.array(a))
        else:
            valid_args.append(a)
    return valid_args[0] if len(valid_args) == 1 else valid_args


def cp_calc(airtemp=scipy.array([]), \
            rh=scipy.array([]), \
            airpress=scipy.array([])):
    """
    Function to calculate the specific heat of air.

    Parameters:
        - airtemp: (array of) air temperature [Celsius].
        - rh: (array of) relative humidity data [%].
        - airpress: (array of) air pressure data [Pa].

    Returns:
        cp: array of saturated c_p values [J kg-1 K-1].

    References
    ----------

    R.G. Allen, L.S. Pereira, D. Raes and M. Smith (1998). Crop
    Evaporation Guidelines for computing crop water requirements,
    FAO - Food and Agriculture Organization of the United Nations.
    Irrigation and drainage paper 56, Chapter 3. Rome, Italy.
    (http://www.fao.org/docrep/x0490e/x0490e07.htm)

    Examples
    --------

        >>> cp_calc(25,60,101300)
        1014.0749457208065
        >>> t = [10, 20, 30]
        >>> rh = [10, 20, 30]
        >>> airpress = [100000, 101000, 102000]
        >>> cp_calc(t,rh,airpress)
        array([ 1005.13411289,  1006.84399787,  1010.83623841])

    """

    # Test input array/value
    airtemp, rh, airpress = convert_to_array(airtemp, rh, airpress)

    # calculate vapour pressures
    eact = ea_calc(airtemp, rh)
    # Calculate cp
    cp = 0.24 * 4185.5 * (1 + 0.8 * (0.622 * eact / (airpress - eact)))
    return cp  # in J/kg/K


def delta_calc(airtemp=scipy.array([])):
    """
    Function to calculate the slope of the temperature - vapour pressure curve
    (Delta) from air temperatures.

    Parameters:
        - airtemp: (array of) air temperature [Celsius].

    Returns:
        - Delta: (array of) slope of saturated vapour curve [Pa K-1].

    References
    ----------

    Technical regulations 49, World Meteorological Organisation, 1984.
    Appendix A. 1-Ap-A-3.

    Examples
    --------
        >>> delta_calc(30.0)
        243.34309166827094
        >>> x = [20, 25]
        >>> delta_calc(x)
        array([ 144.6658414 ,  188.62504569])

    """

    # Test input array/value
    airtemp = convert_to_array(airtemp)

    # calculate saturation vapour pressure at temperature
    es = es_calc(airtemp)  # in Pa
    # Convert es (Pa) to kPa
    es = es / 1000.0
    # Calculate Delta
    delta = es * 4098.0 / ((airtemp + 237.3) ** 2) * 1000
    return delta  # in Pa/K


def ea_calc(airtemp=scipy.array([]), rh=scipy.array([])):
    """
    Function to calculate actual saturation vapour pressure.

    Parameters:
        - airtemp: array of measured air temperatures [Celsius].
        - rh: Relative humidity [%].

    Returns:
        - ea: array of actual vapour pressure [Pa].

    Examples
    --------

        >>> ea_calc(25,60)
        1900.0946514729308

    """

    # Test input array/value
    airtemp, rh = convert_to_array(airtemp, rh)

    # Calculate saturation vapour pressures
    es = es_calc(airtemp)
    # Calculate actual vapour pressure
    eact = rh / 100.0 * es
    return eact  # in Pa


def calc_sat_vapor_press_ice(temperature):
    """
    Calculate the saturation vapor pressure over ice.

    This function computes the saturation vapor pressure over ice
    for a given temperature using the Goff-Gratch equation (1946).

    Parameters:
    -----------
    temperature : float
        Air temperature in degrees Celsius.

    Returns:
    --------
    float
        Saturation vapor pressure over ice in hectopascals (hPa).

    Notes:
    ------
    - This function uses the Goff-Gratch equation for ice, which is considered
      one of the most accurate formulations for saturation vapor pressure over ice.
    - The equation is valid for temperatures below 0°C (273.15 K).
    - The function does not include any error handling for temperatures above freezing.
      For temperatures above 0°C, consider using a function for saturation vapor pressure
      over liquid water instead.

    Formula:
    --------
    The Goff-Gratch equation for ice used in this function is:

    log10(ei) = -9.09718 * (273.16/T - 1)
                - 3.56654 * log10(273.16/T)
                + 0.876793 * (1 - T/273.16)
                + log10(6.1071)

    where:
    - ei is the saturation vapor pressure over ice in hPa
    - T is the absolute temperature in Kelvin

    Example:
    --------
    >>> calc_sat_vapor_press_ice(-10)
    2.5989  # Example output, actual value may differ slightly

    References:
    -----------
    Goff, J. A., and S. Gratch (1946) Low-pressure properties of water from -160 to
    212 F. Transactions of the American Society of Heating and Ventilating Engineers,
    pp 95-122, presented at the 52nd annual meeting of the American Society of Heating
    and Ventilating Engineers, New York, 1946.

    See Also:
    ---------
    calc_sat_vapor_press_water : Function to calculate saturation vapor pressure over liquid water
    """
    # Function implementation...
    log_pi = - 9.09718 * (273.16 / (temperature + 273.15) - 1.0) \
             - 3.56654 * math.log10(273.16 / (temperature + 273.15)) \
             + 0.876793 * (1.0 - (temperature + 273.15) / 273.16) \
             + math.log10(6.1071)
    return math.pow(10, log_pi)


def calc_sat_vapor_press_water(temperature):
    """
    Calculate the saturation vapor pressure over liquid water.

    This function computes the saturation vapor pressure over liquid water
    for a given temperature using the Goff-Gratch equation (1946).

    Parameters:
    -----------
    temperature : float
        Air temperature in degrees Celsius.

    Returns:
    --------
    float
        Saturation vapor pressure over liquid water in hectopascals (hPa).

    Notes:
    ------
    - This function uses the Goff-Gratch equation, which is considered one of the most
      accurate formulations for saturation vapor pressure over liquid water.
    - The equation is valid for temperatures above 0°C (273.15 K).
    - The function does not include any error handling for temperatures below freezing.
      For temperatures below 0°C, consider using a function for saturation vapor pressure
      over ice instead.

    Formula:
    --------
    The Goff-Gratch equation used in this function is:

    log10(ew) = 10.79574 * (1 - 273.16/T)
                - 5.02800 * log10(T/273.16)
                + 1.50475E-4 * (1 - 10^(-8.2969 * (T/273.16 - 1)))
                + 0.42873E-3 * (10^(4.76955 * (1 - 273.16/T)) - 1)
                + 0.78614

    where:
    - ew is the saturation vapor pressure in hPa
    - T is the absolute temperature in Kelvin

    Example:
    --------
    >>> calc_sat_vapor_press_water(20)
    23.3855  # Example output, actual value may differ slightly

    References:
    -----------
    Goff, J. A., and S. Gratch (1946) Low-pressure properties of water from -160 to
    212 F. Transactions of the American Society of Heating and Ventilating Engineers,
    pp 95-122, presented at the 52nd annual meeting of the American Society of Heating
    and Ventilating Engineers, New York, 1946.

    See Also:
    ---------
    calc_sat_vapor_press_ice : Function to calculate saturation vapor pressure over ice
    """

    log_pw = 10.79574 * (1.0 - 273.16 / (temperature + 273.15)) \
             - 5.02800 * math.log10((temperature + 273.15) / 273.16) \
             + 1.50475E-4 * (1 - math.pow(10, (-8.2969 \
                                               * ((temperature + 273.15) / 273.16 - 1.0)))) + 0.42873E-3 \
             * (math.pow(10, (+4.76955 * (1.0 - 273.16 \
                                          / (temperature + 273.15)))) - 1) + 0.78614
    return math.pow(10, log_pw)


def es_calc(airtemp=scipy.array([])):
    """
    Calculate saturation vapor pressure based on air temperature.

    This function computes the saturation vapor pressure for given air temperature(s).
    It handles both single values and arrays, and uses different calculation methods
    for temperatures above and below freezing.

    Parameters:
    -----------
    airtemp : array_like or float, optional
        Air temperature in degrees Celsius. Can be a single value or an array.
        Default is an empty SciPy array.

    Returns:
    --------
    saturation_vapour_pressure : ndarray or float
        Saturation vapor pressure in Pascals (Pa). The shape of the output matches
        the input: a single value for a single input, or an array for array input.

    Notes:
    ------
    - For temperatures below 0°C, saturation vapor pressure over ice is calculated.
    - For temperatures at or above 0°C, saturation vapor pressure over water is calculated.
    - The function uses helper functions `calc_sat_vapor_press_ice` and
      `calc_sat_vapor_press_water` (not shown in the provided code).
    - Input is converted to an array using `convert_to_array` function (not provided).
    - Output is converted from hectopascals (hPa) to pascals (Pa).

    Examples:
    ---------
    >>> es_calc(25)
    3169.2  # Example output, actual value may differ

    >>> es_calc([0, 10, 20, 30])
    array([611.2, 1228.1, 2339.3, 4246.0])  # Example output, actual values may differ

    Raises:
    -------
    TypeError
        If input cannot be converted to a numeric type.

    See Also:
    ---------
    calc_sat_vapor_press_ice : Function to calculate saturation vapor pressure over ice
    calc_sat_vapor_press_water : Function to calculate saturation vapor pressure over water
    """
    airtemp = convert_to_array(airtemp)
    n = scipy.size(airtemp)

    if isinstance(airtemp, np.ndarray):
        saturation_vapour_pressure = scipy.zeros(n)
        for i in range(0, n):
            if airtemp[i] < 0:
                saturation_vapour_pressure[i] = calc_sat_vapor_press_ice(airtemp[i])
            else:
                saturation_vapour_pressure[i] = calc_sat_vapor_press_water(airtemp[i])
    else:
        if airtemp < 0:
            saturation_vapour_pressure = calc_sat_vapor_press_ice(airtemp)
        else:
            saturation_vapour_pressure = calc_sat_vapor_press_water(airtemp)

    saturation_vapour_pressure = saturation_vapour_pressure * 100.0
    return saturation_vapour_pressure


def gamma_calc(airtemp=scipy.array([]), \
               rh=scipy.array([]), \
               airpress=scipy.array([])):
    """
    Function to calculate the psychrometric constant gamma.

    Parameters:
        - airtemp: array of measured air temperature [Celsius].
        - rh: array of relative humidity values[%].
        - airpress: array of air pressure data [Pa].

    Returns:
        - gamma: array of psychrometric constant values [Pa K-1].

    References
    ----------

    J. Bringfelt. Test of a forest evapotranspiration model. Meteorology and
    Climatology Reports 52, SMHI, Norrköpping, Sweden, 1986.

    Examples
    --------

        >>> gamma_calc(10,50,101300)
        66.26343318657227
        >>> t = [10, 20, 30]
        >>> rh = [10, 20, 30]
        >>> airpress = [100000, 101000, 102000]
        >>> gamma_calc(t,rh,airpress)
        array([ 65.25518798,  66.65695779,  68.24239285])

    """

    # Test input array/value
    airtemp, rh, airpress = convert_to_array(airtemp, rh, airpress)

    # Calculate cp and Lambda values
    cp = cp_calc(airtemp, rh, airpress)
    L = le_calc(airtemp)
    # Calculate gamma
    gamma = cp * airpress / (0.622 * L)
    return gamma  # in Pa\K


def le_calc(airtemp=scipy.array([])):
    """
    Function to calculate the latent heat of vapourisation from air temperature.

    Parameters:
        - airtemp: (array of) air temperature [Celsius].

    Returns:
        - L: (array of) lambda [J kg-1 K-1].

    References
    ----------

    J. Bringfelt. Test of a forest evapotranspiration model. Meteorology and
    Climatology Reports 52, SMHI, Norrköpping, Sweden, 1986.

    Examples
    --------

        >>> le_calc(25)
        2440883.8804625
        >>> t=[10, 20, 30]
        >>> le_calc(t)
        array([ 2476387.3842125,  2452718.3817125,  2429049.3792125])

    """

    # Test input array/value
    airtemp = convert_to_array(airtemp)

    # Calculate lambda
    L = 4185.5 * (751.78 - 0.5655 * (airtemp + 273.15))
    return L  # in J/kg


def pottemp(airtemp=scipy.array([]), rh=scipy.array([]), airpress=scipy.array([])):
    """
    Function to calculate the potential temperature air, theta, from air
    temperatures, relative humidity and air pressure. Reference pressure
    1000 hPa.

    Parameters:
        - airtemp: (array of) air temperature data [Celsius].
        - rh: (array of) relative humidity data [%].
        - airpress: (array of) air pressure data [Pa].

    Returns:
        - theta: (array of) potential air temperature data [Celsius].

    Examples
    --------

        >>> t = [5, 10, 20]
        >>> rh = [45, 65, 89]
        >>> airpress = [101300, 102000, 99800]
        >>> pottemp(t,rh,airpress)
        array([  3.97741582,   8.40874555,  20.16596828])
        >>> pottemp(5,45,101300)
        3.977415823848844

    """
    # Test input array/value
    airtemp, rh, airpress = convert_to_array(airtemp, rh, airpress)

    # Determine cp
    cp = cp_calc(airtemp, rh, airpress)
    # Determine theta
    theta = (airtemp + 273.15) * pow((100000.0 / airpress), \
                                     (287.0 / cp)) - 273.15
    return theta  # in degrees celsius


def rho_calc(airtemp=scipy.array([]), \
             rh=scipy.array([]), \
             airpress=scipy.array([])):
    """
    Function to calculate the density of air, rho, from air
    temperatures, relative humidity and air pressure.

    Parameters:
        - airtemp: (array of) air temperature data [Celsius].
        - rh: (array of) relative humidity data [%].
        - airpress: (array of) air pressure data [Pa].

    Returns:
        - rho: (array of) air density data [kg m-3].

    Examples
    --------

        >>> t = [10, 20, 30]
        >>> rh = [10, 20, 30]
        >>> airpress = [100000, 101000, 102000]
        >>> rho_calc(t,rh,airpress)
        array([ 1.22948419,  1.19787662,  1.16635358])
        >>> rho_calc(10,50,101300)
        1.2431927125520903

    """

    # Test input array/value
    airtemp, rh, airpress = convert_to_array(airtemp, rh, airpress)

    # Calculate actual vapour pressure
    eact = ea_calc(airtemp, rh)
    # Calculate density of air rho
    rho = 1.201 * (290.0 * (airpress - 0.378 * eact)) \
          / (1000.0 * (airtemp + 273.15)) / 100.0
    return rho  # in kg/m3


def sun_NR(doy=scipy.array([]), lat=float):
    """
    Function to calculate the maximum sunshine duration [h] and incoming
    radiation [MJ/day] at the top of the atmosphere from day of year and
    latitude.

    Parameters:
     - doy: (array of) day of year.
     - lat: latitude in decimal degrees, negative for southern hemisphere.

    Returns:
    - N: (float, array) maximum sunshine hours [h].
    - Rext: (float, array) extraterrestrial radiation [J day-1].

    Notes
    -----
    Only valid for latitudes between 0 and 67 degrees (i.e. tropics
    and temperate zone).

    References
    ----------

    R.G. Allen, L.S. Pereira, D. Raes and M. Smith (1998). Crop
    Evaporation - Guidelines for computing crop water requirements,
    FAO - Food and Agriculture Organization of the United Nations.
    Irrigation and drainage paper 56, Chapter 3. Rome, Italy.
    (http://www.fao.org/docrep/x0490e/x0490e07.htm)

    Examples
    --------

        >>> sun_NR(50,60)
        (9.1631820597268163, 9346987.824773483)
        >>> days = [100,200,300]
        >>> latitude = 52.
        >>> sun_NR(days,latitude)
        (array([ 13.31552077,  15.87073276,   9.54607624]), array([ 29354803.66244921,  39422316.42084264,  12619144.54566777]))

    """

    # Test input array/value
    doy, lat = convert_to_array(doy, lat)

    # Set solar constant [W/m2]
    S = 1367.0  # [W/m2]
    # Print warning if latitude is above 67 degrees
    if abs(lat) > 67.:
        print
        'WARNING: Latitude outside range of application (0-67 degrees).\n)'
    # Convert latitude [degrees] to radians
    latrad = lat * math.pi / 180.0
    # calculate solar declination dt [radians]
    dt = 0.409 * scipy.sin(2 * math.pi / 365 * doy - 1.39)
    # calculate sunset hour angle [radians]
    ws = scipy.arccos(-scipy.tan(latrad) * scipy.tan(dt))
    # Calculate sunshine duration N [h]
    N = 24 / math.pi * ws
    # Calculate day angle j [radians]
    j = 2 * math.pi / 365.25 * doy
    # Calculate relative distance to sun
    dr = 1.0 + 0.03344 * scipy.cos(j - 0.048869)
    # Calculate Rext
    Rext = S * 86400 / math.pi * dr * (ws * scipy.sin(latrad) * scipy.sin(dt) \
                                       + scipy.sin(ws) * scipy.cos(latrad) * scipy.cos(dt))
    return N, Rext


def vpd_calc(airtemp=scipy.array([]), \
             rh=scipy.array([])):
    """
    Function to calculate vapour pressure deficit.

    Parameters:
        - airtemp: measured air temperatures [Celsius].
        - rh: (array of) rRelative humidity [%].

    Returns:
        - vpd: (array of) vapour pressure deficits [Pa].

    Examples
    --------

        >>> vpd_calc(30,60)
        1697.090397862653
        >>> T=[20,25]
        >>> RH=[50,100]
        >>> vpd_calc(T,RH)
        array([ 1168.54009896,     0.        ])

    """

    # Test input array/value
    airtemp, rh = convert_to_array(airtemp, rh)

    # Calculate saturation vapour pressures
    es = es_calc(airtemp)
    eact = ea_calc(airtemp, rh)
    # Calculate vapour pressure deficit
    vpd = es - eact
    return vpd  # in hPa


def windvec(u=scipy.array([]), D=scipy.array([])):
    """
    Function to calculate the wind vector from time series of wind
    speed and direction.

    Parameters:
        - u: array of wind speeds [m s-1].
        - D: array of wind directions [degrees from North].

    Returns:
        - uv: Vector wind speed [m s-1].
        - Dv: Vector wind direction [degrees from North].

    Examples
    --------

        >>> u = scipy.array([[ 3.],[7.5],[2.1]])
        >>> D = scipy.array([[340],[356],[2]])
        >>> windvec(u,D)
        (4.162354202836905, array([ 353.2118882]))
        >>> uv, Dv = windvec(u,D)
        >>> uv
        4.162354202836905
        >>> Dv
        array([ 353.2118882])

    """

    # Test input array/value
    u, D = convert_to_array(u, D)

    ve = 0.0  # define east component of wind speed
    vn = 0.0  # define north component of wind speed
    D = D * math.pi / 180.0  # convert wind direction degrees to radians
    for i in range(0, len(u)):
        ve = ve + u[i] * math.sin(D[i])  # calculate sum east speed components
        vn = vn + u[i] * math.cos(D[i])  # calculate sum north speed components
    ve = - ve / len(u)  # determine average east speed component
    vn = - vn / len(u)  # determine average north speed component
    uv = math.sqrt(ve * ve + vn * vn)  # calculate wind speed vector magnitude
    # Calculate wind speed vector direction
    vdir = scipy.arctan2(ve, vn)
    vdir = vdir * 180.0 / math.pi  # Convert radians to degrees
    if vdir < 180:
        Dv = vdir + 180.0
    else:
        if vdir > 180.0:
            Dv = vdir - 180
        else:
            Dv = vdir
    return uv, Dv  # uv in m/s, Dv in dgerees from North



def evaplib():
    """
    Evaplib: A library with Python functions for calculation of
    evaporation from meteorological data.
    Evaporation functions
    ---------------------
    - E0: Calculate Penman (1948, 1956) open water evaporation.
    - Em: Calculate evaporation according to Makkink (1965).
    - Ept: Calculate evaporation according to Priestley and Taylor (1972).
    - ET0pm: Calculate Penman Monteith reference evaporation short grass (FAO).
    - Epm: Calculate Penman Monteith reference evaporation (Monteith, 1965).
    - ra: Calculate  from windspeed and roughness parameters.
    - tvardry: calculate sensible heat flux from temperature variations
      (Vugts et al., 1993).
    - gash79: calculate rainfall interception (Gash, 1979).
    Author: Maarten J. Waterloo <m.j.waterloo@vu.nl>.
    Version: 1.0.
    Date: Sep 2012, last modified November 2015.
    """
    details = """
    A library with Python functions for calculation of 
    evaporation from meteorological and vegetation data.
    Functions:
    - E0: Calculate Penman (1948, 1956) (open water) evaporation
    - Em: Calculate evaporation according to Makkink (1965)
    - Ept: Calculate evaporation according to Priestley and Taylor (1972).
    - ET0pm: Calculate Penman Monteith reference evaporation short grass.
    - Epm: Calculate Penman Monteith evaporation (Monteith, 1965).
    - ra: Calculate aerodynamic resistance.
    - tvardry: calculate sensible heat flux from temperature variations 
         (Vugts et al., 1993).
    - gash79: calculate rainfall interception (Gash, 1979).
    """

    AUTHOR = "Maarten J. Waterloo <m.j.waterloo@vu.nl>"
    VERSION = "1.0"
    DATE = "Sep 2012, last modified November 2015"

    print(details)
    print(f"Author: {AUTHOR}")
    print(f"Version: {VERSION}")
    print(f"Date: {DATE}")


# Load meteolib and scientific python modules


def E0(
    airtemp=scipy.array([]),
    rh=scipy.array([]),
    airpress=scipy.array([]),
    Rs=scipy.array([]),
    Rext=scipy.array([]),
    u=scipy.array([]),
    alpha=0.08,
    Z=0.0,
):
    """
    Function to calculate daily Penman (open) water evaporation estimates.

    Parameters:
        airtemp (object):
        - airtemp: (array of) daily average air temperatures [Celsius].
        - rh: (array of) daily average relative humidity [%].
        - airpress: (array of) daily average air pressure data [Pa].
        - Rs: (array of) daily incoming solar radiation [J m-2 day-1].
        - Rext: (array of) daily extraterrestrial radiation [J m-2 day-1].
        - u: (array of) daily average wind speed at 2 m [m s-1].
        - alpha: albedo [-] set at 0.08 for open water by default.
        - Z: (array of) site elevation, default is 0 m a.s.l.

    Returns:
        - E0: (array of) Penman open water evaporation values [mm day-1].

    Notes
    -----

    Meteorological parameters measured at 2 m above the surface. Albedo
    alpha set by default at 0.08 for open water (Valiantzas, 2006).

    References
    ----------

    - H.L. Penman (1948). Natural evaporation from open water, bare soil\
    and grass. Proceedings of the Royal Society of London. Series A.\
    Mathematical and Physical Sciences 193: 120-145.
    - H.L. Penman (1956). Evaporation: An introductory survey. Netherlands\
    Journal of Agricultural Science 4: 9-29.
    - J.D. Valiantzas (2006). Simplified versions for the Penman\
    evaporation equation using routine weather data. J. Hydrology 331:\
    690-702.

    Examples
    --------

        >>> # With single values and default albedo/elevation
        >>> E0(20.67,67.0,101300.0,22600000.,42000000.,3.2)
        6.6029208786994467
        >>> # With albedo is 0.18 instead of default and default elevation
        >>> E0(20.67,67.0,101300.0,22600000.,42000000.,3.2,alpha=0.18)
        5.9664248091431968
        >>> # With standard albedo and Z= 250.0 m
        >>> E0(20.67,67.0,101300.0,22600000.,42000000.,3.2,Z=250.0)
        6.6135588207586284
        >>> # With albedo alpha = 0.18 and elevation Z = 1000 m a.s.l.
        >>> E0(20.67,67.0,101300.0,22600000.,42000000.,3.2,0.18,1000.)
        6.00814764682986

    """

    # Test input array/value
    airtemp, rh, airpress, Rs, Rext, u = convert_to_array(
        airtemp, rh, airpress, Rs, Rext, u
    )

    # Set constants
    sigma = 4.903e-3  # Stefan Boltzmann constant J/m2/K4/d

    # Calculate Delta, gamma and lambda
    DELTA = delta_calc(airtemp)  # [Pa/K]
    gamma = gamma_calc(airtemp, rh, airpress)  # [Pa/K]
    Lambda = le_calc(airtemp)  # [J/kg]

    # Calculate saturated and actual water vapour pressures
    es = es_calc(airtemp)  # [Pa]
    ea = ea_calc(airtemp, rh)  # [Pa]

    # calculate radiation components (J/m2/day)
    Rns = (1.0 - alpha) * Rs  # Shortwave component [J/m2/d]
    Rs0 = (0.75 + 2e-5 * Z) * Rext  # Calculate clear sky radiation Rs0
    f = 1.35 * Rs / Rs0 - 0.35
    epsilom = 0.34 - 0.14 * scipy.sqrt(ea / 1000)
    Rnl = f * epsilom * sigma * (airtemp + 273.15) ** 4  # Longwave component [J/m2/d]
    Rnet = Rns - Rnl  # Net radiation [J/m2/d]
    Ea = (1 + 0.536 * u) * (es / 1000 - ea / 1000)
    E0 = (
        DELTA / (DELTA + gamma) * Rnet / Lambda
        + gamma / (DELTA + gamma) * 6430000 * Ea / Lambda
    )
    return E0


def ET0pm(
    airtemp=scipy.array([]),
    rh=scipy.array([]),
    airpress=scipy.array([]),
    Rs=scipy.array([]),
    Rext=scipy.array([]),
    u=scipy.array([]),
    Z=0.0,
):
    """
    Function to calculate daily Penman Monteith reference evaporation estimates.

    Parameters:
        - airtemp: (array of) daily average air temperatures [Celsius].
        - rh: (array of) daily average relative humidity values [%].
        - airpress: (array of) daily average air pressure data [hPa].
        - Rs: (array of) total incoming shortwave radiation [J m-2 day-1].
        - Rext: Incoming shortwave radiation at the top of the atmosphere\
        [J m-2 day-1].
        - u: windspeed [m s-1].
        - Z: elevation [m], default is 0 m a.s.l.

    Returns:
        - ET0pm: (array of) Penman Monteith reference evaporation (short\
        grass with optimum water supply) values [mm day-1].

    Notes
    -----

    Meteorological measuements standard at 2 m above soil surface.

    References
    ----------

    R.G. Allen, L.S. Pereira, D. Raes and M. Smith (1998). Crop
    evapotranspiration - Guidelines for computing crop water requirements -
    FAO Irrigation and drainage paper 56. FAO - Food and Agriculture
    Organization of the United Nations, Rome, 1998.
    (http://www.fao.org/docrep/x0490e/x0490e07.htm)

    Examples
    --------

        >>> ET0pm(20.67,67.0,101300.0,22600000.,42000000.,3.2)
        4.7235349721073039

    """

    # Test input array/value
    airtemp, rh, airpress, Rs, Rext, u = convert_to_array(
        airtemp, rh, airpress, Rs, Rext, u
    )

    # Set constants
    albedo = 0.23  # short grass albedo
    sigma = 4.903e-3  # Stefan Boltzmann constant J/m2/K4/d

    # Calculate Delta, gamma and lambda
    DELTA = delta_calc(airtemp)  # [Pa/K]
    gamma = gamma_calc(airtemp, rh, airpress)  # [Pa/K]
    Lambda = le_calc(airtemp)  # [J/kg]

    # Calculate saturated and actual water vapour pressures
    es = es_calc(airtemp)  # [Pa]
    ea = ea_calc(airtemp, rh)  # [Pa]

    Rns = (1.0 - albedo) * Rs  # Shortwave component [J/m2/d]
    # Calculate clear sky radiation Rs0
    Rs0 = (0.75 + 2e-5 * Z) * Rext  # Clear sky radiation [J/m2/d]
    f = 1.35 * Rs / Rs0 - 0.35
    epsilom = 0.34 - 0.14 * scipy.sqrt(ea / 1000)
    Rnl = f * epsilom * sigma * (airtemp + 273.15) ** 4  # Longwave component [J/m2/d]
    Rnet = Rns - Rnl  # Net radiation [J/m2/d]
    ET0pm = (
        DELTA / 1000.0 * Rnet / Lambda
        + 900.0 / (airtemp + 273.16) * u * (es - ea) / 1000 * gamma / 1000
    ) / (DELTA / 1000.0 + gamma / 1000 * (1.0 + 0.34 * u))
    return ET0pm  # FAO reference evaporation [mm/day]


def Em(
    airtemp=scipy.array([]),
    rh=scipy.array([]),
    airpress=scipy.array([]),
    Rs=scipy.array([]),
):
    """
    Function to calculate Makkink evaporation (in mm/day).

    The Makkink evaporation is a reference crop evaporation used in the
    Netherlands, which is combined with a crop factor to provide an
    estimate of actual crop evaporation.


    Parameters:
        - airtemp: (array of) daily average air temperatures [Celsius].
        - rh: (array of) daily average relative humidity values [%].
        - airpress: (array of) daily average air pressure data [Pa].
        - Rs: (array of) average daily incoming solar radiation [J m-2 day-1].

    Returns:
        - Em: (array of) Makkink evaporation values [mm day-1].

    Notes
    -----

    Meteorological measuements standard at 2 m above soil surface.

    References
    ----------

    H.A.R. de Bruin (1987). From Penman to Makkink, in Hooghart, C. (Ed.),
    Evaporation and Weather, Proceedings and Information. Comm. Hydrological
    Research TNO, The Hague. pp. 5-30.

    Examples
    --------

        >>> Em(21.65,67.0,101300.,24200000.)
        4.503830479197991

    """

    # Test input array/value
    airtemp, rh, airpress, Rs = convert_to_array(airtemp, rh, airpress, Rs)

    # Calculate Delta and gamma constants
    DELTA = delta_calc(airtemp)
    gamma = gamma_calc(airtemp, rh, airpress)
    Lambda = le_calc(airtemp)

    # calculate Em [mm/day]
    Em = 0.65 * DELTA / (DELTA + gamma) * Rs / Lambda
    return Em


def Ept(
    airtemp=scipy.array([]),
    rh=scipy.array([]),
    airpress=scipy.array([]),
    Rn=scipy.array([]),
    G=scipy.array([]),
):
    """
    Function to calculate daily Priestley - Taylor evaporation.

    Parameters:
        - airtemp: (array of) daily average air temperatures [Celsius].
        - rh: (array of) daily average relative humidity values [%].
        - airpress: (array of) daily average air pressure data [Pa].
        - Rn: (array of) average daily net radiation [J m-2 day-1].
        - G: (array of) average daily soil heat flux [J m-2 day-1].

    Returns:
        - Ept: (array of) Priestley Taylor evaporation values [mm day-1].

    Notes
    -----

    Meteorological parameters normally measured at 2 m above the surface.

    References
    ----------

    Priestley, C.H.B. and R.J. Taylor, 1972. On the assessment of surface
    heat flux and evaporation using large-scale parameters. Mon. Weather
    Rev. 100:81-82.

    Examples
    --------

        >>> Ept(21.65,67.0,101300.,18200000.,600000.)
        6.349456116128078

    """

    # Test input array/value
    airtemp, rh, airpress, Rn, G = convert_to_array(
        airtemp, rh, airpress, Rn, G
    )

    # Calculate Delta and gamma constants
    DELTA = delta_calc(airtemp)
    gamma = gamma_calc(airtemp, rh, airpress)
    Lambda = le_calc(airtemp)
    # calculate Em [mm/day]
    Ept = 1.26 * DELTA / (DELTA + gamma) * (Rn - G) / Lambda
    return Ept


def ra(z=float, z0=float, d=float, u=scipy.array([])):
    """
    Function to calculate aerodynamic resistance.

    Parameters:
        - z: measurement height [m].
        - z0: roughness length [m].
        - d: displacement length [m].
        - u: (array of) windspeed [m s-1].

    Returns:
        - ra: (array of) aerodynamic resistances [s m-1].

    References
    ----------

    A.S. Thom (1075), Momentum, mass and heat exchange of plant communities,
    In: Monteith, J.L. Vegetation and the Atmosphere, Academic Press, London.
    p. 57–109.

    Examples
    --------

        >>> ra(3,0.12,2.4,5.0)
        3.2378629924752942
        >>> u=([2,4,6])
        >>> ra(3,0.12,2.4,u)
        array([ 8.09465748,  4.04732874,  2.69821916])

    """

    # Test input array/value
    u = convert_to_array(u)

    # Calculate ra
    ra = (scipy.log((z - d) / z0)) ** 2 / (0.16 * u)
    return ra  # aerodynamic resistanc in s/m


def Delta_calc(T):
    """
    Calculate the slope of the saturation vapor pressure curve (Delta) at a given temperature.

    Args:
        T (numpy.ndarray or float): Temperature in degrees Celsius

    Returns:
        numpy.ndarray or float: Slope of the vapor pressure curve (kPa °C^-1)

    Notes:
        - Formula based on FAO-56 methodology
        - Valid for temperatures between -20°C and 50°C
        - Returns np.nan for temperatures outside valid range

    Reference:
        Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998.
        FAO Irrigation and Drainage Paper No. 56.
    """

    # Convert input to numpy array if it isn't already
    T = np.asarray(T)

    # Create mask for valid temperature range
    valid_mask = np.logical_and(T >= -20, T <= 50)

    # Initialize output array with nans
    Delta = np.full_like(T, np.nan, dtype=float)

    # Calculate saturation vapor pressure (es)
    # es = 0.6108 * exp((17.27 * T)/(T + 237.3))
    es = 0.6108 * np.exp((17.27 * T[valid_mask]) / (T[valid_mask] + 237.3))

    # Calculate slope of vapor pressure curve (Delta)
    # Delta = 4098 * es / (T + 237.3)^2
    Delta[valid_mask] = (4098 * es) / ((T[valid_mask] + 237.3) ** 2)

    # If input was a scalar, return scalar
    if np.isscalar(T):
        return float(Delta)

    return Delta


def L_calc(T):
    """
    Calculate the latent heat of vaporization (λ) for water at given temperature(s).

    Args:
        T (numpy.ndarray or float): Temperature in degrees Celsius

    Returns:
        numpy.ndarray or float: Latent heat of vaporization (MJ kg^-1)

    Notes:
        - Formula based on Harrison (1963)
        - Valid for temperatures between 0°C and 100°C
        - Returns np.nan for temperatures outside valid range
        - At 20°C, λ ≈ 2.45 MJ kg^-1

    Reference:
        Harrison, L.P., 1963. Fundamental concepts and definitions relating to
        humidity. In: Wexler, A. (Ed.), Humidity and Moisture, Vol. 3.
    """

    # Convert input to numpy array if it isn't already
    T = np.asarray(T)

    # Create mask for valid temperature range
    valid_mask = np.logical_and(T >= 0, T <= 100)

    # Initialize output array with nans
    L = np.full_like(T, np.nan, dtype=float)

    # Calculate latent heat of vaporization
    # L = 2.501 - (2.361e-3 * T)  # Result in MJ kg^-1
    L[valid_mask] = 2.501 - (2.361e-3 * T[valid_mask])

    # If input was a scalar, return scalar
    if np.isscalar(T):
        return float(L)

    return L


def Epm(
    airtemp=scipy.array([]),
    rh=scipy.array([]),
    airpress=scipy.array([]),
    Rn=scipy.array([]),
    G=scipy.array([]),
    ra=scipy.array([]),
    rs=scipy.array([]),
):
    """
    Function to calculate the Penman Monteith evaporation.

    Parameters:
        - airtemp: (array of) daily average air temperatures [Celsius].
        - rh: (array of) daily average relative humidity values [%].
        - airpress: (array of) daily average air pressure data [hPa].
        - Rn: (array of) average daily net radiation [J].
        - G: (array of) average daily soil heat flux [J].
        - ra: aerodynamic resistance [s m-1].
        - rs: surface resistance [s m-1].

    Returns:
        - Epm: (array of) Penman Monteith evaporation values [mm].

    References
    ----------

    J.L. Monteith (1965) Evaporation and environment. Symp. Soc. Exp. Biol.
    19, 205-224.

    Examples
    --------

        >>> Epm(21.67,67.0,101300.0,10600000.,500000.0,11.0,120.)
        6.856590956174142

    """

    # Calculate Delta, gamma and lambda
    DELTA = Delta_calc(airtemp) / 100.0  # [hPa/K]
    airpress = airpress * 100.0  # [Pa]
    gamma = gamma_calc(airtemp, rh, airpress) / 100.0  # [hPa/K]
    Lambda = L_calc(airtemp)  # [J/kg]
    rho = rho_calc(airtemp, rh, airpress)
    cp = cp_calc(airtemp, rh, airpress)
    # Calculate saturated and actual water vapour pressures
    es = es_calc(airtemp) / 100.0  # [hPa]
    ea = ea_calc(airtemp, rh) / 100.0  # [hPa]
    # Calculate Epm
    Epm = (
        DELTA * Rn + rho * cp * (es - ea) * ra / (DELTA + gamma * (1.0 + rs / ra))
    ) / Lambda
    return Epm  # actual ET in mm


def tvardry(
    rho=scipy.array([]),
    cp=scipy.array([]),
    T=scipy.array([]),
    sigma_t=scipy.array([]),
    z=float(),
    d=0.0,
):
    """Function to calculate the sensible heat flux from high
    frequency temperature measurements and its standard deviation.

    Parameters:
        - rho: (array of) air density values [kg m-3].
        - cp: (array of) specific heat at constant temperature values [J kg-1 K-1].
        - T: (array of) temperature data [Celsius].
        - sigma_t: (array of) standard deviation of temperature data [Celsius].
        - z: temperature measurement height above the surface [m].
        - d: displacement height due to vegetation, default is zero [m].

    Returns:
        - H: (array of) sensible heat flux [W m-2].

    Notes
    -----
    This function holds only for free convective conditions when C2*z/L >>1,
    where L is the Obhukov length.

    References
    ----------
    - J.E. Tillman (1972), The indirect determination of stability, heat and\
    momentum fluxes in the atmosphere boundary layer from simple scalar\
    variables during dry unstable conditions, Journal of Applied Meteorology\
    11: 783-792.
    - H.F. Vugts, M.J. Waterloo, F.J. Beekman, K.F.A. Frumau and L.A.\
    Bruijnzeel. The temperature variance method: a powerful tool in the\
    estimation of actual evaporation rates. In J. S. Gladwell, editor,\
    Hydrology of Warm Humid Regions, Proc. of the Yokohama Symp., IAHS\
    Publication No. 216, pages 251-260, July 1993.

    Examples
    --------

        >>> tvardry(1.25,1035.0,25.3,0.25,3.0)
        34.658669290185287
        >>> d=0.25
        >>> tvardry(1.25,1035.0,25.3,0.25,3.0,d)
        33.183149497185511

    """

    # Define constants
    k = 0.40  # von Karman constant
    g = 9.81  # acceleration due to gravity [m/s^2]
    C1 = 2.9  # De Bruin et al., 1992
    C2 = 28.4  # De Bruin et al., 1992
    # L= Obhukov-length [m]

    # Free Convection Limit
    H = rho * cp * scipy.sqrt((sigma_t / C1) ** 3 * k * g * (z - d) / (T + 273.15) * C2)
    # else:
    # including stability correction
    # zoverL = z/L
    # tvardry = rho * cp * scipy.sqrt((sigma_t/C1)**3 * k*g*(z-d) / (T+273.15) *\
    #          (1-C2*z/L)/(-1*z/L))

    # Check if we get complex numbers (square root of negative value) and remove
    # I = find(zoL >= 0 | H.imag != 0);
    # H(I) = scipy.ones(size(I))*NaN;

    return H  # sensible heat flux


def calculate_canopy_interception(
        Pg, ER, S, p, pt, PGsat, canopy_storage=0.0, trunk_storage=0.0
):
    if Pg < PGsat and Pg > 0:
        canopy_storage = (1 - p - pt) * Pg
        if Pg > canopy_storage / pt:
            trunk_storage = canopy_storage + pt * Pg
    if Pg > PGsat:
        canopy_storage = (((1 - p - pt) * PGsat) - S) + (ER * (Pg - PGsat)) + S
        if Pg > (canopy_storage / pt):
            canopy_storage += canopy_storage + pt * Pg
            trunk_storage = canopy_storage + pt * Pg
    return canopy_storage, trunk_storage


def calculate_rainfall_series(Pg, ER, S, p, pt, PGsat, Ei, TF):
    n = scipy.size(Pg)
    for i in range(0, n):
        Ecan, Etrunk = calculate_canopy_interception(Pg[i], ER, S, p, pt, PGsat)
        Ei[i] = Ecan + Etrunk
        TF[i] = Pg[i] - Ei[i]


def gash79(Pg=scipy.array([]), ER=float, S=float, canopy_storage=float, pt=float):
    """
    Function to calculate precipitation interception loss.
    """
    rainfall_length = scipy.size(Pg)

    # Check if we have a single precipitation value or an array
    if rainfall_length < 2:  # Dealing with single value...
        # PGsat calculation (for the saturation of the canopy)
        PGsat = -(1 / ER * S) * scipy.log((1 - (ER / (1 - canopy_storage - pt))))
        Ecan, Etrunk = calculate_canopy_interception(
            Pg, ER, S, canopy_storage, pt, PGsat
        )
        Ei = Ecan + Etrunk
        TF = Pg - Ei
        SF = 0
    else:
        # Define variables and constants
        Ei = scipy.zeros(rainfall_length)
        TF = scipy.zeros(rainfall_length)
        SF = scipy.zeros(rainfall_length)
        PGsat = -(1 / ER * S) * scipy.log((1 - (ER / (1 - canopy_storage - pt))))

        calculate_rainfall_series(Pg, ER, S, canopy_storage, pt, PGsat, Ei, TF)

    return Pg, TF, SF, Ei



