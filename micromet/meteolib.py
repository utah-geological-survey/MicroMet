
"""
Improved Meteorological Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A modern Python library for meteorological calculations.

"""

from typing import Union, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GRAVITY = 9.81  # Acceleration due to gravity [m/s^2]
VON_KARMAN = 0.41  # von Karman constant
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]


def validate_inputs(**kwargs) -> None:
    """Validate input parameters"""
    for name, value in kwargs.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            if not all(np.isfinite(x) for x in np.asarray(value).flatten()):
                raise MeteoError(f"Invalid {name}: contains non-finite values")
        elif not np.isfinite(value):
            raise MeteoError(f"Invalid {name}: {value} is not finite")

def to_array(*args):
    """Convert inputs to numpy arrays while preserving single values"""
    results = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            results.append(np.array(arg))
        else:
            results.append(arg)
    return results[0] if len(results) == 1 else results


class TemperatureUnit(Enum):
    """Temperature unit enumeration"""
    CELSIUS = "C"
    KELVIN = "K"
    FAHRENHEIT = "F"

@dataclass
class MeteoConfig:
    """Configuration parameters for meteorological calculations"""
    temp_unit: TemperatureUnit = TemperatureUnit.CELSIUS
    validate_inputs: bool = True
    raise_warnings: bool = True


class MeteoError(Exception):
    """Base exception class for meteorological calculation errors"""
    pass



@dataclass
class SolarResults:
    """Container for solar calculation results"""
    max_sunshine_hours: Union[float, np.ndarray]
    extraterrestrial_radiation: Union[float, np.ndarray]

@dataclass
class MeteoConfig:
    """Configuration parameters for meteorological calculations"""
    temp_unit: TemperatureUnit = TemperatureUnit.CELSIUS
    validate_inputs: bool = True
    raise_warnings: bool = True

class MeteoError(Exception):
    """Base exception class for meteorological calculation errors"""
    pass

class MeteoCalculator:
    """
    Modern implementation of meteorological calculations.

    Features:
    - Type checking
    - Input validation
    - Vectorized operations
    - Comprehensive error handling
    """

    def __init__(self, config: Optional[MeteoConfig] = None):
        """Initialize calculator with optional configuration"""
        self.config = config or MeteoConfig()

    @staticmethod
    def _validate_inputs(**kwargs) -> None:
        """
        Validate meteorological input parameters.

        Args:
            **kwargs: Keyword arguments containing parameters to validate.
                Recognized parameters:
                - airtemp: Air temperature [°C]
                - rh: Relative humidity [%]
                - airpress: Air pressure [Pa]

        Raises:
            MeteoError: If any input parameter is invalid
        """
        # Physical limits
        LIMITS = {
            'airtemp': (-273.15, 100),  # Physical minimum to reasonable maximum
            'rh': (0, 100),  # Physical range for relative humidity
            'airpress': (1000, 120000)  # From high mountains to deepest valleys
        }

        for name, value in kwargs.items():
            # Skip if None (optional parameter)
            if value is None:
                continue

            # Convert to numpy array for consistent handling
            value = np.asarray(value)

            # Check for non-finite values
            if np.any(~np.isfinite(value)):
                raise MeteoError(f"Invalid {name}: contains non-finite values")

            # Check physical limits if defined
            if name in LIMITS:
                min_val, max_val = LIMITS[name]
                if np.any(value < min_val) or np.any(value > max_val):
                    raise MeteoError(
                        f"Invalid {name}: must be between {min_val} and {max_val}, "
                        f"got {value}"
                    )

    def specific_heat(self,
                      airtemp: Union[float, np.ndarray],
                      rh: Union[float, np.ndarray],
                      airpress: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate specific heat of air.

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            airpress: Air pressure [Pa]

        Returns:
            Specific heat [J kg⁻¹ K⁻¹]
        """
        # Convert inputs to numpy arrays
        airtemp = np.asarray(airtemp)
        rh = np.asarray(rh)
        airpress = np.asarray(airpress)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

        # Calculate vapor pressures
        eact = self.actual_vapor_pressure(airtemp, rh)

        # Vectorized calculation
        cp = 0.24 * 4185.5 * (1 + 0.8 * (0.622 * eact / (airpress - eact)))
        return cp

    def vapor_pressure_slope(self,
                             airtemp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate slope of temperature-vapor pressure curve.

        Args:
            airtemp: Air temperature [°C]

        Returns:
            Slope [Pa K⁻¹]
        """
        airtemp = np.asarray(airtemp)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp)

        # Calculate saturation vapor pressure
        es = self.saturation_vapor_pressure(airtemp)
        es_kpa = es / 1000.0

        # Vectorized calculation using numpy
        delta = es_kpa * 4098.0 / ((airtemp + 237.3) ** 2) * 1000
        return delta

    def saturation_vapor_pressure(self, airtemp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate saturation vapor pressure using the Buck equation (1981).

        This formulation is simpler than Goff-Gratch but still highly accurate
        (within 0.05% of full Goff-Gratch equation).

        Args:
            airtemp: Air temperature [°C]

        Returns:
            Saturation vapor pressure [Pa]

        References:
            Buck, A.L., 1981: New equations for computing vapor pressure and
            enhancement factor. J. Appl. Meteorol., 20, 1527-1532.

        Warns:
            UserWarning: If temperature is outside recommended range (-80 to 50°C)
        """

        airtemp = np.asarray(airtemp)

        # Input validation and warnings
        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp)

        # Check temperature range and issue warning
        TEMP_MIN = -80
        TEMP_MAX = 50
        if np.any(airtemp < TEMP_MIN) or np.any(airtemp > TEMP_MAX):
            warnings.warn(
                f"Temperature {airtemp}°C is outside recommended range "
                f"({TEMP_MIN} to {TEMP_MAX}°C). Results may be inaccurate.",
                UserWarning
            )

        # Constants for Buck equation
        a_water = 17.502
        b_water = 240.97
        a_ice = 22.587
        b_ice = 273.86
        es0 = 611.21  # Reference vapor pressure at 0°C

        # For better numerical stability around 0°C, use a smooth transition
        # between ice and water formulations
        weight = 1 / (1 + np.exp(-2 * airtemp))  # Sigmoid function

        # Calculate both ice and water formulations
        es_water = es0 * np.exp(a_water * airtemp / (b_water + airtemp))
        es_ice = es0 * np.exp(a_ice * airtemp / (b_ice + airtemp))

        # Blend the two formulations smoothly
        es = es_ice * (1 - weight) + es_water * weight

        return es

    @staticmethod
    def _saturation_vapor_pressure_ice(temp: np.ndarray) -> np.ndarray:
        """Calculate saturation vapor pressure over ice"""
        T = temp + 273.15
        return np.exp((-9.09718 * (273.16 / T - 1.0)
                       - 3.56654 * np.log10(273.16 / T)
                       + 0.876793 * (1.0 - T / 273.16)
                       + np.log10(6.1071)))

    @staticmethod
    def _saturation_vapor_pressure_water(temp: np.ndarray) -> np.ndarray:
        """Calculate saturation vapor pressure over water"""
        T = temp + 273.15
        return np.exp((10.79574 * (1.0 - 273.16 / T)
                       - 5.02800 * np.log10(T / 273.16)
                       + 1.50475e-4 * (1 - 10 ** (-8.2969 * (T / 273.16 - 1.0)))
                       + 0.42873e-3 * (10 ** (4.76955 * (1.0 - 273.16 / T)) - 1)
                       + 0.78614))

    def wind_vector(self,
                    speed: Union[float, np.ndarray],
                    direction: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray],
    Union[float, np.ndarray]]:
        """
        Calculate wind vector from speed and direction.

        Args:
            speed: Wind speed [m s⁻¹]
            direction: Wind direction [degrees from North]

        Returns:
            Tuple of (vector speed [m s⁻¹], vector direction [degrees])
        """
        speed = np.asarray(speed)
        direction = np.asarray(direction)

        if self.config.validate_inputs:
            self._validate_inputs(speed=speed, direction=direction)

        # Convert direction to radians
        dir_rad = np.radians(direction)

        # Calculate vector components
        ve = -np.mean(speed * np.sin(dir_rad))
        vn = -np.mean(speed * np.cos(dir_rad))

        # Calculate magnitude and direction
        magnitude = np.sqrt(ve ** 2 + vn ** 2)
        vector_dir = np.degrees(np.arctan2(ve, vn))

        # Adjust direction
        vector_dir = np.where(vector_dir < 180,
                              vector_dir + 180,
                              np.where(vector_dir > 180,
                                       vector_dir - 180,
                                       vector_dir))

        return magnitude, vector_dir

    def actual_vapor_pressure(self,
                              airtemp: Union[float, np.ndarray],
                              rh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate actual vapor pressure from air temperature and relative humidity.

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]

        Returns:
            Actual vapor pressure [Pa]

        Examples:
            >>> calc = MeteoCalculator()
            >>> calc.actual_vapor_pressure(25, 60)
            1900.09
        """
        airtemp = np.asarray(airtemp)
        rh = np.asarray(rh)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp, rh=rh)
            if np.any((rh < 0) | (rh > 100)):
                raise MeteoError("Relative humidity must be between 0-100%")

        # Calculate saturation vapor pressure and actual vapor pressure
        es = self.saturation_vapor_pressure(airtemp)
        ea = rh / 100.0 * es

        return ea

    def psychrometric_constant(self,
                               airtemp: Union[float, np.ndarray],
                               rh: Union[float, np.ndarray],
                               airpress: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the psychrometric constant.

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            airpress: Air pressure [Pa]

        Returns:
            Psychrometric constant [Pa K⁻¹]

        References:
            Bringfelt (1986)
        """
        airtemp = np.asarray(airtemp)
        rh = np.asarray(rh)
        airpress = np.asarray(airpress)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

        # Calculate specific heat and latent heat
        cp = self.specific_heat(airtemp, rh, airpress)
        L = self.latent_heat(airtemp)

        # Calculate psychrometric constant
        gamma = cp * airpress / (0.622 * L)

        return gamma

    def latent_heat(self,
                    airtemp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate latent heat of vaporization for water.

        Args:
            airtemp: Air temperature [°C]

        Returns:
            Latent heat of vaporization [J kg⁻¹]

        Notes:
            Valid for temperature range -40 to +40 °C
        """
        airtemp = np.asarray(airtemp)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp)
            if np.any((airtemp < -40) | (airtemp > 40)):
                warnings.warn("Temperature outside recommended range (-40 to 40°C)")

        # Vectorized calculation
        L = 4185.5 * (751.78 - 0.5655 * (airtemp + 273.15))

        return L

    def potential_temperature(self,
                              airtemp: Union[float, np.ndarray],
                              rh: Union[float, np.ndarray],
                              airpress: Union[float, np.ndarray],
                              ref_press: float = 100000.0) -> Union[float, np.ndarray]:
        """
        Calculate potential temperature referenced to a pressure level.

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            airpress: Air pressure [Pa]
            ref_press: Reference pressure level [Pa], default 100000 Pa (1000 hPa)

        Returns:
            Potential temperature [°C]
        """
        airtemp = np.asarray(airtemp)
        rh = np.asarray(rh)
        airpress = np.asarray(airpress)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

        # Calculate specific heat
        cp = self.specific_heat(airtemp, rh, airpress)

        # Convert temperature to Kelvin for calculation
        T = airtemp + 273.15

        # Calculate potential temperature using Poisson's equation
        theta = T * (ref_press / airpress) ** (287.0 / cp)

        # Convert back to Celsius
        return theta - 273.15

    def air_density(self,
                    airtemp: Union[float, np.ndarray],
                    rh: Union[float, np.ndarray],
                    airpress: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate air density using the ideal gas law with moisture correction.

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            airpress: Air pressure [Pa]

        Returns:
            Air density [kg m⁻³]
        """
        airtemp = np.asarray(airtemp)
        rh = np.asarray(rh)
        airpress = np.asarray(airpress)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

        # Calculate actual vapor pressure
        ea = self.actual_vapor_pressure(airtemp, rh)

        # Convert temperature to Kelvin
        T = airtemp + 273.15

        # Calculate density with moisture correction
        rho = ((airpress - 0.378 * ea) / (287.05 * T))

        return rho

    def solar_parameters(self,
                         doy: Union[float, np.ndarray],
                         lat: float) -> SolarResults:
        """
        Calculate maximum sunshine duration and extraterrestrial radiation.

        Args:
            doy: Day of year [1-366]
            lat: Latitude [degrees], negative for Southern hemisphere

        Returns:
            SolarResults object containing:
                - max_sunshine_hours: Maximum possible sunshine duration [hours]
                - extraterrestrial_radiation: Radiation at top of atmosphere [J day⁻¹]

        Notes:
            Valid for latitudes between -67° and +67°
        """
        doy = np.asarray(doy)

        if self.config.validate_inputs:
            self._validate_inputs(doy=doy)
            if abs(lat) > 67:
                warnings.warn("Latitude outside valid range (-67° to +67°)")
            if np.any((doy < 1) | (doy > 366)):
                raise MeteoError("Day of year must be between 1 and 366")

        # Convert latitude to radians
        lat_rad = np.radians(lat)

        # Solar constant [W m⁻²]
        S0 = 1367.0

        # Calculate solar declination [radians]
        decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)

        # Calculate sunset hour angle [radians]
        ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))

        # Calculate maximum sunshine duration [hours]
        N = 24 / np.pi * ws

        # Calculate relative distance to sun
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.25)

        # Calculate extraterrestrial radiation [J day⁻¹]
        Ra = S0 * 86400 / np.pi * dr * (
                ws * np.sin(lat_rad) * np.sin(decl) +
                np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )

        return SolarResults(N, Ra)

    def vapor_pressure_deficit(self,
                               airtemp: Union[float, np.ndarray],
                               rh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate vapor pressure deficit.

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]

        Returns:
            Vapor pressure deficit [Pa]
        """
        airtemp = np.asarray(airtemp)
        rh = np.asarray(rh)

        if self.config.validate_inputs:
            self._validate_inputs(airtemp=airtemp, rh=rh)

        # Calculate saturation and actual vapor pressures
        es = self.saturation_vapor_pressure(airtemp)
        ea = self.actual_vapor_pressure(airtemp, rh)

        # Calculate deficit
        vpd = es - ea

        return vpd

    def penman_monteith_reference(self,
                                  airtemp: Union[float, np.ndarray],
                                  rh: Union[float, np.ndarray],
                                  airpress: Union[float, np.ndarray],
                                  rs: Union[float, np.ndarray],
                                  rn: Union[float, np.ndarray],
                                  g: Union[float, np.ndarray],
                                  u2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Penman-Monteith reference evapotranspiration

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            airpress: Air pressure [Pa]
            rs: Surface resistance [s m⁻¹]
            rn: Net radiation [W m⁻²]
            g: Ground heat flux [W m⁻²]
            u2: Wind speed at 2m height [m s⁻¹]

        Returns:
            Reference ET [mm day⁻¹]
        """
        # Convert inputs to arrays
        inputs = to_array(airtemp, rh, airpress, rs, rn, g, u2)
        airtemp, rh, airpress, rs, rn, g, u2 = inputs

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress,
                            rs=rs, rn=rn, g=g, u2=u2)

        # Calculate required parameters
        delta = self.vapor_pressure_slope(airtemp)
        gamma = self.psychrometric_constant(airtemp, rh, airpress)
        lambda_e = self.latent_heat(airtemp)
        vpd = self.vapor_pressure_deficit(airtemp, rh)
        rho = self.air_density(airtemp, rh, airpress)
        cp = self.specific_heat(airtemp, rh, airpress)

        # Calculate aerodynamic resistance
        ra = 208.0 / u2  # simplified for grass reference

        # Calculate ET using Penman-Monteith
        num = delta * (rn - g) + rho * cp * vpd / ra
        den = delta + gamma * (1 + rs / ra)
        et = num / (lambda_e * den)

        return et * 86400  # Convert to mm/day

    def calculate_radiation(self,
                            airtemp: Union[float, np.ndarray],
                            rh: Union[float, np.ndarray],
                            rs: Union[float, np.ndarray],
                            lat: float,
                            doy: int,
                            albedo: float = 0.23) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate radiation components

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            rs: Incoming solar radiation [W m⁻²]
            lat: Latitude [degrees]
            doy: Day of year
            albedo: Surface albedo [-]

        Returns:
            Dict containing radiation components [W m⁻²]
        """
        airtemp, rh, rs = to_array(airtemp, rh, rs)

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, rs=rs)

        # Calculate extraterrestrial radiation
        ra = self._extraterrestrial_radiation(lat, doy)

        # Net shortwave
        rns = (1 - albedo) * rs

        # Net longwave using improved formulation
        ea = self.actual_vapor_pressure(airtemp, rh)
        tk = airtemp + 273.15
        rnl = STEFAN_BOLTZMANN * tk ** 4 * (0.34 - 0.14 * np.sqrt(ea / 1000)) * \
              (1.35 * rs / ra - 0.35)

        # Net radiation
        rn = rns - rnl

        return {
            'extraterrestrial': ra,
            'net_shortwave': rns,
            'net_longwave': rnl,
            'net_radiation': rn
        }

    def _extraterrestrial_radiation(self, lat: float, doy: int) -> float:
        """Calculate extraterrestrial radiation"""
        lat_rad = np.radians(lat)

        # Solar declination
        decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

        # Inverse relative distance Earth-Sun
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

        # Sunset hour angle
        ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))

        # Extraterrestrial radiation
        ra = 24 * 60 / np.pi * 0.082 * dr * \
             (ws * np.sin(lat_rad) * np.sin(decl) + \
              np.cos(lat_rad) * np.cos(decl) * np.sin(ws))

        return ra

    def penman_open_water(self,
                          airtemp: Union[float, np.ndarray],
                          rh: Union[float, np.ndarray],
                          u2: Union[float, np.ndarray],
                          airpress: Union[np.ndarray, float],
                          rn: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Penman open water evaporation

        Args:
            airtemp: Air temperature [°C]
            rh: Relative humidity [%]
            u2: Wind speed at 2m [m s⁻¹]
            rn: Net radiation [W m⁻²]

        Returns:
            Open water evaporation [mm day⁻¹]
        """
        airtemp, rh, u2, rn = to_array(airtemp, rh, u2, rn)

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, u2=u2, rn=rn)

        # Calculate parameters
        delta = self.vapor_pressure_slope(airtemp)
        gamma = self.psychrometric_constant(airtemp, rh, airpress)
        lambda_e = self.latent_heat(airtemp)
        ea = self.actual_vapor_pressure(airtemp, rh)
        es = self.saturation_vapor_pressure(airtemp)

        # Wind function
        fu = 2.6 * (1 + 0.54 * u2)

        # Calculate evaporation
        erad = delta * rn / lambda_e
        evap = gamma * fu * (es - ea) / lambda_e

        e0 = (erad + evap) / (delta + gamma)

        return e0 * 86400  # Convert to mm/day

    def E0(self, airtemp: Union[float, np.ndarray],
           rh: Union[float, np.ndarray],
           airpress: Union[float, np.ndarray],
           Rs: Union[float, np.ndarray],
           Rext: Union[float, np.ndarray],
           u: Union[float, np.ndarray],
           alpha: float = 0.08,
           Z: float = 0.0) -> Union[float, np.ndarray]:
        """
        Calculate Penman (1948, 1956) open water evaporation.

        Args:
            airtemp: Daily average air temperatures [°C]
            rh: Daily average relative humidity [%]
            airpress: Daily average air pressure [Pa]
            Rs: Daily incoming solar radiation [J m-2 day-1]
            Rext: Daily extraterrestrial radiation [J m-2 day-1]
            u: Daily average wind speed at 2m [m s-1]
            alpha: Albedo [-], default 0.08 for open water
            Z: Site elevation [m]

        Returns:
            Open water evaporation [mm day-1]
        """
        airtemp, rh, airpress, Rs, Rext, u = map(np.asarray,
                                                 [airtemp, rh, airpress, Rs, Rext, u])

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress,
                            Rs=Rs, Rext=Rext, u=u)

        # Calculate parameters
        Delta = self.vapor_pressure_slope(airtemp)
        gamma = self.psychrometric_constant(airtemp, rh, airpress)
        Lambda = self.latent_heat(airtemp)
        es = self.saturation_vapor_pressure(airtemp)
        ea = self.actual_vapor_pressure(airtemp, rh)

        # Calculate radiation components
        Rns = (1.0 - alpha) * Rs
        Rs0 = (0.75 + 2e-5 * Z) * Rext
        f = 1.35 * Rs / Rs0 - 0.35
        epsilon = 0.34 - 0.14 * np.sqrt(ea / 1000)
        Rnl = f * epsilon * STEFAN_BOLTZMANN * (airtemp + 273.15) ** 4
        Rnet = Rns - Rnl

        # Calculate evaporation terms
        Ea = (1 + 0.536 * u) * (es / 1000 - ea / 1000)
        E0 = (Delta / (Delta + gamma) * Rnet / Lambda +
              gamma / (Delta + gamma) * 6430000 * Ea / Lambda)

        return E0

    def Em(self, airtemp: Union[float, np.ndarray],
           rh: Union[float, np.ndarray],
           airpress: Union[float, np.ndarray],
           Rs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Makkink (1965) evaporation.

        Args:
            airtemp: Daily average air temperatures [°C]
            rh: Daily average relative humidity [%]
            airpress: Daily average air pressure [Pa]
            Rs: Average daily incoming solar radiation [J m-2 day-1]

        Returns:
            Makkink evaporation [mm day-1]
        """
        airtemp, rh, airpress, Rs = map(np.asarray, [airtemp, rh, airpress, Rs])

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress, Rs=Rs)

        Delta = self.vapor_pressure_slope(airtemp)
        gamma = self.psychrometric_constant(airtemp, rh, airpress)
        Lambda = self.latent_heat(airtemp)

        return 0.65 * Delta / (Delta + gamma) * Rs / Lambda

    def Ept(self, airtemp: Union[float, np.ndarray],
            rh: Union[float, np.ndarray],
            airpress: Union[float, np.ndarray],
            Rn: Union[float, np.ndarray],
            G: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Priestley-Taylor (1972) evaporation.

        Args:
            airtemp: Daily average air temperatures [°C]
            rh: Daily average relative humidity [%]
            airpress: Daily average air pressure [Pa]
            Rn: Average daily net radiation [J m-2 day-1]
            G: Average daily soil heat flux [J m-2 day-1]

        Returns:
            Priestley-Taylor evaporation [mm day-1]
        """
        airtemp, rh, airpress, Rn, G = map(np.asarray,
                                           [airtemp, rh, airpress, Rn, G])

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress, Rn=Rn, G=G)

        Delta = self.vapor_pressure_slope(airtemp)
        gamma = self.psychrometric_constant(airtemp, rh, airpress)
        Lambda = self.latent_heat(airtemp)

        return 1.26 * Delta / (Delta + gamma) * (Rn - G) / Lambda

    def ET0pm(self, airtemp: Union[float, np.ndarray],
              rh: Union[float, np.ndarray],
              airpress: Union[float, np.ndarray],
              Rs: Union[float, np.ndarray],
              Rext: Union[float, np.ndarray],
              u: Union[float, np.ndarray],
              Z: float = 0.0) -> Union[float, np.ndarray]:
        """
        Calculate FAO Penman-Monteith reference evaporation for short grass.

        Args:
            airtemp: Daily average air temperatures [°C]
            rh: Daily average relative humidity [%]
            airpress: Daily average air pressure [Pa]
            Rs: Daily incoming solar radiation [J m-2 day-1]
            Rext: Extraterrestrial radiation [J m-2 day-1]
            u: Wind speed at 2m [m s-1]
            Z: Elevation [m]

        Returns:
            Reference evapotranspiration [mm day-1]
        """
        airtemp, rh, airpress, Rs, Rext, u = map(np.asarray,
                                                 [airtemp, rh, airpress, Rs, Rext, u])

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress,
                            Rs=Rs, Rext=Rext, u=u)

        # Constants for short grass
        albedo = 0.23

        # Calculate parameters
        Delta = self.vapor_pressure_slope(airtemp)
        gamma = self.psychrometric_constant(airtemp, rh, airpress)
        Lambda = self.latent_heat(airtemp)
        es = self.saturation_vapor_pressure(airtemp)
        ea = self.actual_vapor_pressure(airtemp, rh)

        # Calculate radiation terms
        Rns = (1.0 - albedo) * Rs
        Rs0 = (0.75 + 2e-5 * Z) * Rext
        f = 1.35 * Rs / Rs0 - 0.35
        epsilon = 0.34 - 0.14 * np.sqrt(ea / 1000)
        Rnl = f * epsilon * STEFAN_BOLTZMANN * (airtemp + 273.15) ** 4
        Rnet = Rns - Rnl

        # Calculate ET0
        ET0 = ((Delta / 1000.0 * Rnet / Lambda +
                900.0 / (airtemp + 273.16) * u * (es - ea) / 1000 * gamma / 1000) /
               (Delta / 1000.0 + gamma / 1000 * (1.0 + 0.34 * u)))

        return ET0

    def Epm(self, airtemp: Union[float, np.ndarray],
            rh: Union[float, np.ndarray],
            airpress: Union[float, np.ndarray],
            Rn: Union[float, np.ndarray],
            G: Union[float, np.ndarray],
            ra: Union[float, np.ndarray],
            rs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Penman-Monteith evaporation (Monteith, 1965).

        Args:
            airtemp: Daily average air temperatures [°C]
            rh: Daily average relative humidity [%]
            airpress: Daily average air pressure [Pa]
            Rn: Average daily net radiation [J]
            G: Average daily soil heat flux [J]
            ra: Aerodynamic resistance [s m-1]
            rs: Surface resistance [s m-1]

        Returns:
            Actual evapotranspiration [mm]
        """
        airtemp, rh, airpress, Rn, G, ra, rs = map(np.asarray,
                                                   [airtemp, rh, airpress, Rn, G, ra, rs])

        if self.config.validate_inputs:
            validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress,
                            Rn=Rn, G=G, ra=ra, rs=rs)

        # Calculate parameters
        Delta = self.vapor_pressure_slope(airtemp) / 100.0  # [hPa/K]
        gamma = self.psychrometric_constant(airtemp, rh, airpress) / 100.0
        Lambda = self.latent_heat(airtemp)
        rho = self.air_density(airtemp, rh, airpress)
        cp = self.specific_heat(airtemp, rh, airpress)
        es = self.saturation_vapor_pressure(airtemp) / 100.0
        ea = self.actual_vapor_pressure(airtemp, rh) / 100.0

        # Calculate evaporation
        Epm = ((Delta * Rn + rho * cp * (es - ea) / ra) /
               (Delta + gamma * (1.0 + rs / ra))) / Lambda

        return Epm

    def ra(self, z: float, z0: float, d: float,
           u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate aerodynamic resistance from wind speed and roughness parameters.

        Args:
            z: Measurement height [m]
            z0: Roughness length [m]
            d: Displacement length [m]
            u: Wind speed [m s-1]

        Returns:
            Aerodynamic resistance [s m-1]
        """
        u = np.asarray(u)

        if self.config.validate_inputs:
            if z <= (d + z0):
                raise ValueError("Measurement height must be greater than d + z0")
            validate_inputs(u=u)

        return (np.log((z - d) / z0)) ** 2 / (0.16 * u)

    def tvardry(self, rho: Union[float, np.ndarray],
                cp: Union[float, np.ndarray],
                T: Union[float, np.ndarray],
                sigma_t: Union[float, np.ndarray],
                z: float,
                d: float = 0.0) -> Union[float, np.ndarray]:
        """
        Calculate sensible heat flux from temperature variations.

        Args:
            rho: Air density [kg m-3]
            cp: Specific heat at constant temperature [J kg-1 K-1]
            T: Temperature [°C]
            sigma_t: Standard deviation of temperature [°C]
            z: Temperature measurement height [m]
            d: Displacement height [m]

        Returns:
            Sensible heat flux [W m-2]
        """
        rho, cp, T, sigma_t = map(np.asarray, [rho, cp, T, sigma_t])

        if self.config.validate_inputs:
            validate_inputs(rho=rho, cp=cp, T=T, sigma_t=sigma_t)

        # Constants from De Bruin et al., 1992
        C1 = 2.9
        C2 = 28.4

        # Calculate sensible heat flux
        H = (rho * cp * np.sqrt((sigma_t / C1) ** 3 * VON_KARMAN * GRAVITY *
                                (z - d) / (T + 273.15) * C2))

        return H

    def gash79(self,
               Pg: Union[float, np.ndarray],
               ER: float,
               S: float,
               p: float,
               pt: float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray],
    Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Calculate rainfall interception using Gash (1979) analytical model.

        Args:
            Pg: Gross precipitation [mm]
            ER: Mean evaporation rate [mm/hr]
            S: Canopy storage capacity [mm]
            p: Free throughfall coefficient [-]
            pt: Stemflow coefficient [-]

        Returns:
            Tuple containing:
            - Pg: Gross precipitation [mm]
            - TF: Throughfall [mm]
            - SF: Stemflow [mm]
            - Ei: Interception loss [mm]

        References:
            Gash, J.H.C. (1979). An analytical model of rainfall interception by forests.
            Quarterly Journal of the Royal Meteorological Society, 105: 43-55.

        Notes:
            - The model assumes that evaporation occurs at a constant rate (ER)
            - p + pt should not exceed 1.0 (100% of precipitation)
            - Canopy storage capacity (S) must be positive
        """
        # Convert input to numpy array
        Pg = np.asarray(Pg)

        # Input validation
        if self.config.validate_inputs:
            validate_inputs(Pg=Pg, ER=ER, S=S)

            if not 0 <= p <= 1:
                raise ValueError("Free throughfall coefficient (p) must be between 0 and 1")
            if not 0 <= pt <= 1:
                raise ValueError("Stemflow coefficient (pt) must be between 0 and 1")
            if p + pt > 1:
                raise ValueError("Sum of p and pt must not exceed 1")
            if S <= 0:
                raise ValueError("Canopy storage capacity must be positive")
            if ER <= 0:
                raise ValueError("Evaporation rate must be positive")

        # Initialize output arrays
        rainfall_length = np.size(Pg)
        if rainfall_length < 2:
            # Single value case
            # Calculate saturation point (amount of rainfall needed to saturate canopy)
            PGsat = -(S / ER) * np.log((1 - (ER / (1 - p - pt))))

            # Calculate storages
            if Pg < PGsat and Pg > 0:
                # Case 1: Rainfall insufficient to saturate canopy
                canopy_storage = (1 - p - pt) * Pg
                trunk_storage = 0
                if Pg > canopy_storage / pt:
                    trunk_storage = pt * Pg
            else:
                # Case 2: Rainfall sufficient to saturate canopy
                if Pg > 0:
                    canopy_storage = ((1 - p - pt) * PGsat - S) + (ER * (Pg - PGsat)) + S
                    if Pg > (canopy_storage / pt):
                        trunk_storage = pt * Pg
                    else:
                        trunk_storage = 0
                else:
                    canopy_storage = 0
                    trunk_storage = 0

            # Calculate components
            Ei = canopy_storage + trunk_storage
            TF = Pg - Ei
            SF = 0

        else:
            # Array case
            Ei = np.zeros(rainfall_length)
            TF = np.zeros(rainfall_length)
            SF = np.zeros(rainfall_length)
            PGsat = -(S / ER) * np.log((1 - (ER / (1 - p - pt))))

            # Calculate for each timestep
            for i in range(rainfall_length):
                if Pg[i] < PGsat and Pg[i] > 0:
                    # Insufficient rainfall to saturate canopy
                    canopy_storage = (1 - p - pt) * Pg[i]
                    trunk_storage = 0
                    if Pg[i] > canopy_storage / pt:
                        trunk_storage = pt * Pg[i]
                else:
                    # Sufficient rainfall to saturate canopy
                    if Pg[i] > 0:
                        canopy_storage = ((1 - p - pt) * PGsat - S) + (ER * (Pg[i] - PGsat)) + S
                        if Pg[i] > (canopy_storage / pt):
                            trunk_storage = pt * Pg[i]
                        else:
                            trunk_storage = 0
                    else:
                        canopy_storage = 0
                        trunk_storage = 0

                Ei[i] = canopy_storage + trunk_storage
                TF[i] = Pg[i] - Ei[i]

        # Log warnings for potentially problematic values
        if self.config.raise_warnings:
            if np.any(Ei < 0):
                logger.warning("Negative interception values detected")
            if np.any(TF < 0):
                logger.warning("Negative throughfall values detected")
            if np.any(Ei > Pg):
                logger.warning("Interception exceeds gross precipitation")

        return Pg, TF, SF, Ei

    def _validate_gash_parameters(self, Pg: np.ndarray, ER: float, S: float, p: float, pt: float) -> None:
        """Helper method to validate Gash model parameters"""
        try:
            if np.any(Pg < 0):
                raise ValueError("Gross precipitation cannot be negative")
            if ER <= 0:
                raise ValueError("Mean evaporation rate must be positive")
            if S <= 0:
                raise ValueError("Canopy storage capacity must be positive")
            if not 0 <= p <= 1:
                raise ValueError("Free throughfall coefficient must be between 0 and 1")
            if not 0 <= pt <= 1:
                raise ValueError("Stemflow coefficient must be between 0 and 1")
            if p + pt > 1:
                raise ValueError("Sum of free throughfall and stemflow coefficients cannot exceed 1")
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            raise


# Example usage:
if __name__ == "__main__":
    calc = MeteoCalculator()

    # Example calculations
    temp = 25.0
    rh = 60.0
    pressure = 101300.0

    cp = calc.specific_heat(temp, rh, pressure)
    print(f"Specific heat: {cp:.2f} J kg⁻¹ K⁻¹")

    delta = calc.vapor_pressure_slope(temp)
    print(f"Vapor pressure slope: {delta:.2f} Pa K⁻¹")