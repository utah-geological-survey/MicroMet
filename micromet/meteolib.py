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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print('Ran all tests...')
