# -*- coding: utf-8 -*-
"""
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

__author__ = "Maarten J. Waterloo <maarten.waterloo@falw.vu.nl>"
__version__ = "1.0"
__date__ = "November 2014"

import scipy
import meteolib


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
    airtemp, rh, airpress, Rs, Rext, u = meteolib.convert_to_array(
        airtemp, rh, airpress, Rs, Rext, u
    )

    # Set constants
    sigma = 4.903e-3  # Stefan Boltzmann constant J/m2/K4/d

    # Calculate Delta, gamma and lambda
    DELTA = meteolib.delta_calc(airtemp)  # [Pa/K]
    gamma = meteolib.gamma_calc(airtemp, rh, airpress)  # [Pa/K]
    Lambda = meteolib.le_calc(airtemp)  # [J/kg]

    # Calculate saturated and actual water vapour pressures
    es = meteolib.es_calc(airtemp)  # [Pa]
    ea = meteolib.ea_calc(airtemp, rh)  # [Pa]

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
    airtemp, rh, airpress, Rs, Rext, u = meteolib.convert_to_array(
        airtemp, rh, airpress, Rs, Rext, u
    )

    # Set constants
    albedo = 0.23  # short grass albedo
    sigma = 4.903e-3  # Stefan Boltzmann constant J/m2/K4/d

    # Calculate Delta, gamma and lambda
    DELTA = meteolib.delta_calc(airtemp)  # [Pa/K]
    gamma = meteolib.gamma_calc(airtemp, rh, airpress)  # [Pa/K]
    Lambda = meteolib.le_calc(airtemp)  # [J/kg]

    # Calculate saturated and actual water vapour pressures
    es = meteolib.es_calc(airtemp)  # [Pa]
    ea = meteolib.ea_calc(airtemp, rh)  # [Pa]

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
    airtemp, rh, airpress, Rs = meteolib.convert_to_array(airtemp, rh, airpress, Rs)

    # Calculate Delta and gamma constants
    DELTA = meteolib.delta_calc(airtemp)
    gamma = meteolib.gamma_calc(airtemp, rh, airpress)
    Lambda = meteolib.le_calc(airtemp)

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
    airtemp, rh, airpress, Rn, G = meteolib.convert_to_array(
        airtemp, rh, airpress, Rn, G
    )

    # Calculate Delta and gamma constants
    DELTA = meteolib.delta_calc(airtemp)
    gamma = meteolib.gamma_calc(airtemp, rh, airpress)
    Lambda = meteolib.le_calc(airtemp)
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
    u = meteolib.convert_to_array(u)

    # Calculate ra
    ra = (scipy.log((z - d) / z0)) ** 2 / (0.16 * u)
    return ra  # aerodynamic resistanc in s/m


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

    # Test input array/value
    airtemp, rh, airpress, Rn, G, ra, rs = meteolib._arraytest(
        airtemp, rh, airpress, Rn, G, ra, rs
    )

    # Calculate Delta, gamma and lambda
    DELTA = meteolib.Delta_calc(airtemp) / 100.0  # [hPa/K]
    airpress = airpress * 100.0  # [Pa]
    gamma = meteolib.gamma_calc(airtemp, rh, airpress) / 100.0  # [hPa/K]
    Lambda = meteolib.L_calc(airtemp)  # [J/kg]
    rho = meteolib.rho_calc(airtemp, rh, airpress)
    cp = meteolib.cp_calc(airtemp, rh, airpress)
    # Calculate saturated and actual water vapour pressures
    es = meteolib.es_calc(airtemp) / 100.0  # [hPa]
    ea = meteolib.ea_calc(airtemp, rh) / 100.0  # [hPa]
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

    # Test input array/value
    rho, cp, T, sigma_t = meteolib._arraytest(rho, cp, T, sigma_t)

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


# Run doctest when executing module

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print("Ran all tests...")
