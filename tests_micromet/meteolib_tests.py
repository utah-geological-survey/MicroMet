"""
Test suite for MeteoCalculator class.
Uses pytest for testing framework.

Coverage includes:
- Input validation
- Edge cases
- Physical bounds
- Array operations
- Known reference values
"""

import pytest
import numpy as np
from micromet.meteolib import MeteoCalculator, MeteoError, MeteoConfig, TemperatureUnit
import warnings

@pytest.fixture
def calc():
    """Create MeteoCalculator instance for testing"""
    return MeteoCalculator()

# Test data fixture with typical meteorological values
# Test data fixture with typical meteorological values
@pytest.fixture
def test_data():
    """Provide typical meteorological values for testing"""
    return {
        'airtemp': 25.0,           # Air temperature [°C]
        'rh': 60.0,                # Relative humidity [%]
        'airpress': 101300.0,      # Air pressure [Pa]
        'Rs': 20e6,                # Solar radiation [J m-2 day-1]
        'Rext': 40e6,              # Extraterrestrial radiation [J m-2 day-1]
        'Rn': 15e6,                # Net radiation [J m-2 day-1]
        'G': 1e6,                  # Ground heat flux [J m-2 day-1]
        'u': 2.5,                  # Wind speed [m s-1]
        'Z': 100.0,                # Elevation [m]
        'ra': 50.0,                # Aerodynamic resistance [s m-1]
        'rs': 70.0                 # Surface resistance [s m-1]
    }


class TestPenmanEvaporation:
    """Test suite for Penman (1948, 1956) open water evaporation"""

    def test_E0_invalid_inputs(self, calc):
        """Test Penman E0 with invalid inputs"""
        # Test temperature below absolute zero
        with pytest.raises(MeteoError, match=r"Invalid airtemp: must be between -273.15 and 100"):
            calc.E0(-300.0, 60.0, 101300.0, 20e6, 40e6, 2.5)

        # Test invalid relative humidity
        with pytest.raises(MeteoError, match=r"Invalid rh: must be between 0 and 100"):
            calc.E0(25.0, 150.0, 101300.0, 20e6, 40e6, 2.5)

        # Test invalid air pressure
        with pytest.raises(MeteoError, match=r"Invalid airpress: must be between 1000 and 120000"):
            calc.E0(25.0, 60.0, 0.0, 20e6, 40e6, 2.5)

        # Test non-finite values
        with pytest.raises(MeteoError, match=r"Invalid airtemp: contains non-finite values"):
            calc.E0(float('nan'), 60.0, 101300.0, 20e6, 40e6, 2.5)

    def test_E0_single_values(self, calc, test_data):
        """Test Penman E0 with single values"""
        E0 = calc.E0(test_data['airtemp'],
                     test_data['rh'],
                     test_data['airpress'],
                     test_data['Rs'],
                     test_data['Rext'],
                     test_data['u'])

        assert isinstance(E0, float)
        assert E0 > 0
        assert E0 < 15  # Reasonable daily evaporation limit [mm/day]

    def test_E0_array_inputs(self, calc):
        """Test Penman E0 with array inputs"""
        airtemp = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 60.0, 70.0])
        airpress = np.array([101300.0, 101300.0, 101300.0])
        Rs = np.array([18e6, 20e6, 22e6])
        Rext = np.array([35e6, 40e6, 42e6])
        u = np.array([2.0, 2.5, 3.0])

        E0 = calc.E0(airtemp, rh, airpress, Rs, Rext, u)

        assert isinstance(E0, np.ndarray)
        assert len(E0) == 3
        assert np.all(E0 > 0)
        assert np.all(E0 < 15)

    def test_E0_boundary_values(self, calc):
        """Test Penman E0 with boundary values"""
        # Test at absolute zero (should work)
        E0_cold = calc.E0(-273.15, 60.0, 101300.0, 20e6, 40e6, 2.5)
        assert isinstance(E0_cold, float)
        assert E0_cold >= 0

        # Test at maximum temperature (should work)
        E0_hot = calc.E0(100.0, 60.0, 101300.0, 20e6, 40e6, 2.5)
        assert isinstance(E0_hot, float)
        assert E0_hot > E0_cold  # Higher temperature should give higher evaporation


class TestMakkinkEvaporation:
    """Test suite for Makkink (1965) evaporation"""

    def test_Em_single_values(self, calc, test_data):
        """Test Makkink Em with single values"""
        Em = calc.Em(test_data['airtemp'],
                     test_data['rh'],
                     test_data['airpress'],
                     test_data['Rs'])

        assert isinstance(Em, float)
        assert Em > 0
        assert Em < 10  # Reasonable daily evaporation limit

    def test_Em_array_inputs(self, calc):
        """Test Makkink Em with array inputs"""
        airtemp = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 60.0, 70.0])
        airpress = np.array([101300.0, 101300.0, 101300.0])
        Rs = np.array([18e6, 20e6, 22e6])

        Em = calc.Em(airtemp, rh, airpress, Rs)

        assert isinstance(Em, np.ndarray)
        assert len(Em) == 3
        assert np.all(Em > 0)
        assert np.all(Em < 10)

    def test_Em_comparison_with_E0(self, calc, test_data):
        """Test that Makkink Em is typically less than Penman E0"""
        Em = calc.Em(test_data['airtemp'],
                     test_data['rh'],
                     test_data['airpress'],
                     test_data['Rs'])

        E0 = calc.E0(test_data['airtemp'],
                     test_data['rh'],
                     test_data['airpress'],
                     test_data['Rs'],
                     test_data['Rext'],
                     test_data['u'])

        assert Em < E0


class TestPriestleyTaylorEvaporation:
    """Test suite for Priestley-Taylor (1972) evaporation"""

    def test_Ept_single_values(self, calc, test_data):
        """Test Priestley-Taylor Ept with single values"""
        Ept = calc.Ept(test_data['airtemp'],
                       test_data['rh'],
                       test_data['airpress'],
                       test_data['Rn'],
                       test_data['G'])

        assert isinstance(Ept, float)
        assert Ept > 0
        assert Ept < 12  # Reasonable daily evaporation limit

    def test_Ept_array_inputs(self, calc):
        """Test Priestley-Taylor Ept with array inputs"""
        airtemp = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 60.0, 70.0])
        airpress = np.array([101300.0, 101300.0, 101300.0])
        Rn = np.array([12e6, 15e6, 17e6])
        G = np.array([0.8e6, 1.0e6, 1.2e6])

        Ept = calc.Ept(airtemp, rh, airpress, Rn, G)

        assert isinstance(Ept, np.ndarray)
        assert len(Ept) == 3
        assert np.all(Ept > 0)
        assert np.all(Ept < 12)

    def test_Ept_zero_heat_flux(self, calc, test_data):
        """Test Priestley-Taylor Ept with zero ground heat flux"""
        data = test_data.copy()
        data['G'] = 0.0

        Ept = calc.Ept(data['airtemp'],
                       data['rh'],
                       data['airpress'],
                       data['Rn'],
                       data['G'])

        assert Ept > 0


class TestPenmanMonteithEvaporation:
    """Test suite for Penman-Monteith evaporation"""

    def test_Epm_single_values(self, calc, test_data):
        """Test Penman-Monteith Epm with single values"""
        Epm = calc.Epm(test_data['airtemp'],
                       test_data['rh'],
                       test_data['airpress'],
                       test_data['Rn'],
                       test_data['G'],
                       test_data['ra'],
                       test_data['rs'])

        assert isinstance(Epm, float)
        assert Epm > 0
        assert Epm < 15  # Reasonable daily evaporation limit

    def test_Epm_array_inputs(self, calc):
        """Test Penman-Monteith Epm with array inputs"""
        airtemp = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 60.0, 70.0])
        airpress = np.array([101300.0, 101300.0, 101300.0])
        Rn = np.array([12e6, 15e6, 17e6])
        G = np.array([0.8e6, 1.0e6, 1.2e6])
        ra = np.array([45.0, 50.0, 55.0])
        rs = np.array([65.0, 70.0, 75.0])

        Epm = calc.Epm(airtemp, rh, airpress, Rn, G, ra, rs)

        assert isinstance(Epm, np.ndarray)
        assert len(Epm) == 3
        assert np.all(Epm > 0)
        assert np.all(Epm < 15)

    def test_Epm_zero_resistance(self, calc, test_data):
        """Test Penman-Monteith Epm with zero surface resistance"""
        data = test_data.copy()
        data['rs'] = 0.0

        Epm = calc.Epm(data['airtemp'],
                       data['rh'],
                       data['airpress'],
                       data['Rn'],
                       data['G'],
                       data['ra'],
                       data['rs'])

        assert Epm > 0


class TestFAOPenmanMonteithEvaporation:
    """Test suite for FAO Penman-Monteith reference evaporation"""

    def test_ET0pm_single_values(self, calc, test_data):
        """Test FAO Penman-Monteith ET0pm with single values"""
        ET0 = calc.ET0pm(test_data['airtemp'],
                         test_data['rh'],
                         test_data['airpress'],
                         test_data['Rs'],
                         test_data['Rext'],
                         test_data['u'])

        assert isinstance(ET0, float)
        assert ET0 > 0
        assert ET0 < 12  # Reasonable daily reference ET limit

    def test_ET0pm_array_inputs(self, calc):
        """Test FAO Penman-Monteith ET0pm with array inputs"""
        airtemp = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 60.0, 70.0])
        airpress = np.array([101300.0, 101300.0, 101300.0])
        Rs = np.array([18e6, 20e6, 22e6])
        Rext = np.array([35e6, 40e6, 42e6])
        u = np.array([2.0, 2.5, 3.0])

        ET0 = calc.ET0pm(airtemp, rh, airpress, Rs, Rext, u)

        assert isinstance(ET0, np.ndarray)
        assert len(ET0) == 3
        assert np.all(ET0 > 0)
        assert np.all(ET0 < 12)

    def test_ET0pm_elevation_effects(self, calc, test_data):
        """Test FAO Penman-Monteith ET0pm response to elevation changes"""
        ET0_low = calc.ET0pm(test_data['airtemp'],
                             test_data['rh'],
                             test_data['airpress'],
                             test_data['Rs'],
                             test_data['Rext'],
                             test_data['u'],
                             Z=0.0)

        ET0_high = calc.ET0pm(test_data['airtemp'],
                              test_data['rh'],
                              test_data['airpress'],
                              test_data['Rs'],
                              test_data['Rext'],
                              test_data['u'],
                              Z=2000.0)

        assert ET0_high != ET0_low  # Elevation should affect results


class TestEvaporationComparisons:
    """Test suite for comparing different evaporation methods"""

    def test_method_relationships(self, calc, test_data):
        """Test expected relationships between different evaporation methods"""
        # Calculate evaporation using different methods
        E0 = calc.E0(test_data['airtemp'],
                     test_data['rh'],
                     test_data['airpress'],
                     test_data['Rs'],
                     test_data['Rext'],
                     test_data['u'])

        Em = calc.Em(test_data['airtemp'],
                     test_data['rh'],
                     test_data['airpress'],
                     test_data['Rs'])

        ET0 = calc.ET0pm(test_data['airtemp'],
                         test_data['rh'],
                         test_data['airpress'],
                         test_data['Rs'],
                         test_data['Rext'],
                         test_data['u'])

        # Check expected relationships
        assert E0 > Em  # Penman open water should be higher than Makkink
        assert E0 > ET0  # Open water evaporation should exceed reference ET
        assert Em < E0 * 1.5  # Makkink shouldn't be unreasonably high compared to Penman

    def test_extreme_conditions(self, calc):
        """Test evaporation methods under extreme but valid conditions"""
        # Hot, dry conditions - based on realistic extreme desert conditions
        hot_dry = {
            'airtemp': 45.0,  # Death Valley-like maximum
            'rh': 15.0,  # Very low humidity
            'airpress': 101300.0,
            'Rs': 35e6,  # Maximum realistic solar radiation
            'Rext': 45e6,
            'u': 5.0
        }

        # Cold, humid conditions - based on realistic cool temperate conditions
        cold_humid = {
            'airtemp': 5.0,
            'rh': 90.0,
            'airpress': 101300.0,
            'Rs': 8e6,  # Low winter radiation
            'Rext': 20e6,
            'u': 1.0
        }

        # Test both conditions
        for conditions in [hot_dry, cold_humid]:
            E0 = calc.E0(**conditions)
            ET0 = calc.ET0pm(**conditions)

            assert E0 > 0
            assert ET0 > 0

            # Maximum theoretical limits based on energy balance and atmospheric demand
            # For reference: highest recorded ET0 values are around 20-22 mm/day
            # in extremely hot, dry, and windy desert conditions
            assert E0 < 30  # Open water evaporation can be higher than ET0
            assert ET0 < 22  # Based on maximum observed values in extreme desert conditions

            # Check that cold conditions produce lower evaporation
            if conditions['airtemp'] < 10:
                assert E0 < 5  # Low evaporation expected in cold conditions
                assert ET0 < 3  # Even lower reference ET in cold conditions

    def test_elevation_effects(self, calc, test_data):
        """Test the effect of elevation on evaporation estimates"""
        # Calculate ET0 at different elevations
        ET0_sea_level = calc.ET0pm(
            test_data['airtemp'],
            test_data['rh'],
            101325.0,  # Standard sea level pressure
            test_data['Rs'],
            test_data['Rext'],
            test_data['u'],
            Z=0.0
        )

        # Pressure at ~2000m elevation using standard atmosphere approximation
        p_2000m = 101325.0 * (1 - 2000 / 44330.0) ** 5.255

        ET0_mountain = calc.ET0pm(
            test_data['airtemp'],
            test_data['rh'],
            p_2000m,
            test_data['Rs'],
            test_data['Rext'],
            test_data['u'],
            Z=2000.0
        )

        # Higher elevation should increase ET0 due to lower air pressure
        # and typically higher radiation, but difference shouldn't be extreme
        assert ET0_mountain > ET0_sea_level
        assert ET0_mountain < ET0_sea_level * 1.5  # Reasonable limit on elevation effect

    def test_radiation_limits(self, calc, test_data):
        """Test evaporation estimates with extreme radiation values"""

        # Test with very high radiation (extreme desert conditions)
        high_rad = test_data.copy()
        high_rad['Rs'] = 35e6  # Very high solar radiation
        high_rad['Rext'] = 45e6

        E0_high = calc.E0(high_rad['airtemp'],
                          high_rad['rh'],
                          high_rad['airpress'],
                          high_rad['Rs'],
                          high_rad['Rext'],
                          high_rad['u'],
                          )

        assert E0_high < 25  # But should still be physically reasonable

        # Test with very low radiation (nighttime conditions)
        low_rad = test_data.copy()
        low_rad['Rs'] = 1e5  # Very low solar radiation
        low_rad['Rext'] = 1e6

        E0_low = calc.E0(low_rad['airtemp'],
                         low_rad['rh'],
                         low_rad['airpress'],
                         low_rad['Rs'],
                         low_rad['Rext'],
                         low_rad['u'],)
        assert E0_low >= 0  # Evaporation should not be negative
        assert E0_low < 3  # Low radiation should result in very low evaporation
        assert E0_high > E0_low  # Higher radiation should increase evaporation

class TestInputValidation:
    """Tests for input validation"""

    def test_invalid_temperature(self, calc):
        """Test handling of invalid temperature inputs"""
        with pytest.raises(MeteoError):
            calc.specific_heat(np.nan, 60, 101300)

        with pytest.raises(MeteoError):
            calc.specific_heat(np.inf, 60, 101300)

    def test_invalid_humidity(self, calc):
        """Test handling of invalid humidity inputs"""
        with pytest.raises(MeteoError):
            calc.actual_vapor_pressure(20, -10)  # Negative RH

        with pytest.raises(MeteoError):
            calc.actual_vapor_pressure(20, 110)  # RH > 100%

    def test_valid_pressure(self, calc):
        """Test acceptance of valid pressure inputs"""
        valid_pressures = [
            101325,  # Standard pressure
            85000,  # Mountain pressure
            [101325, 85000]  # Array of valid pressures
        ]

        for pressure in valid_pressures:
            try:
                if isinstance(pressure, (list, tuple)):
                    pressure = np.array(pressure)
                calc._validate_inputs(airpress=pressure)
            except MeteoError as e:
                pytest.fail(f"Valid pressure {pressure} raised MeteoError: {e}")

    def test_multiple_invalid_inputs(self, calc):
        """Test handling of multiple invalid inputs simultaneously"""
        with pytest.raises(MeteoError):
            calc._validate_inputs(
                airtemp=1000,  # Too hot
                rh=-10,  # Invalid RH
                airpress=-1000  # Invalid pressure
            )

    def test_invalid_day_of_year(self, calc):
        """Test handling of invalid day of year inputs"""
        with pytest.raises(MeteoError):
            calc.solar_parameters(0, 45)  # Day < 1

        with pytest.raises(MeteoError):
            calc.solar_parameters(367, 45)  # Day > 366


class TestVaporPressure:
    """Tests for vapor pressure calculations"""

    def test_saturation_vapor_pressure_reference_values(self, calc):
        """Test saturation vapor pressure calculation against known reference values"""
        # Reference values from Smithsonian Meteorological Tables
        reference_values = {
            -20: 103.2,  # Below freezing
            -10: 259.9,
            0: 611.2,  # Freezing point
            10: 1227.9,
            20: 2338.7,
            30: 4246.0,
            40: 7384.9
        }

        for temp, expected in reference_values.items():
            result = calc.saturation_vapor_pressure(temp)
            assert np.isclose(result, expected, rtol=1e-3), \
                f"Failed at {temp}°C: got {result:.1f} Pa, expected {expected} Pa"

    def test_saturation_vapor_pressure_array(self, calc):
        """Test array inputs for saturation vapor pressure"""
        temps = np.array([-10, 0, 10, 20])
        expected = np.array([259.9, 611.2, 1227.9, 2338.7])
        result = calc.saturation_vapor_pressure(temps)

        assert np.allclose(result, expected, rtol=1e-3)
        assert result.shape == temps.shape

    def test_saturation_vapor_pressure_physical(self, calc):
        """Test physical properties of saturation vapor pressure"""
        # Test across a range avoiding exact zero for better numerical stability
        temps = np.linspace(-50, 50, 101)
        es = calc.saturation_vapor_pressure(temps)

        # Should be strictly increasing
        assert np.all(np.diff(es) > 0)

        # Should be positive
        assert np.all(es > 0)

        # Test convexity in separate regions to avoid issues at phase transition
        def test_convexity(t_range):
            es_subset = calc.saturation_vapor_pressure(t_range)
            d2es = np.diff(np.diff(es_subset))
            return np.all(d2es > 0)

        # Test convexity for temperatures well below and above freezing
        assert test_convexity(np.linspace(-50, -5, 50))  # Ice phase
        assert test_convexity(np.linspace(5, 50, 50))  # Liquid phase

        # Test continuity around freezing point
        temps_transition = np.linspace(-1, 1, 201)
        es_transition = calc.saturation_vapor_pressure(temps_transition)

        # Should be continuous (smooth first derivative)
        d1 = np.diff(es_transition)
        assert np.all(d1 > 0)  # Still monotonic
        assert np.allclose(np.diff(d1), 0, atol=1e-2)  # Approximately smooth

    def test_saturation_vapor_pressure_reference(self, calc):
        """Test against reference values"""
        reference = {
            -20: 103.2,
            -10: 259.9,
            0: 611.2,
            10: 1227.9,
            20: 2338.7,
            30: 4246.0,
            40: 7384.9
        }

        for temp, expected in reference.items():
            result = calc.saturation_vapor_pressure(temp)
            assert np.isclose(result, expected, rtol=1e-3), \
                f"At {temp}°C: expected {expected} Pa, got {result} Pa"

    def test_saturation_vapor_pressure_extremes(self, calc):
        """Test behavior at extreme temperatures"""
        # Test very cold temperature
        with pytest.warns(UserWarning, match="outside recommended range"):
            very_cold = calc.saturation_vapor_pressure(-100)
            assert very_cold > 0
            assert np.isfinite(very_cold)

        # Test very hot temperature
        with pytest.warns(UserWarning, match="outside recommended range"):
            very_hot = calc.saturation_vapor_pressure(60)
            assert very_hot > calc.saturation_vapor_pressure(59)
            assert np.isfinite(very_hot)

        # Test array of extreme temperatures
        with pytest.warns(UserWarning, match="outside recommended range"):
            extremes = calc.saturation_vapor_pressure(np.array([-90, 70]))
            assert np.all(np.isfinite(extremes))
            assert np.all(extremes > 0)

    def test_normal_range_no_warning(self, calc):
        """Test that no warning is issued for normal temperature range"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            normal = calc.saturation_vapor_pressure(20)  # Should not warn
            assert np.isfinite(normal)
            assert normal > 0

    def test_saturation_vapor_pressure_freezing(self, calc):
        """Test behavior around freezing point"""
        # Test slightly above and below freezing
        temps = np.array([-0.1, 0.0, 0.1])
        results = calc.saturation_vapor_pressure(temps)

        # Should be continuous at freezing
        assert np.isclose(results[1], 611.2, rtol=1e-3)
        # Should be monotonic
        assert np.all(np.diff(results) > 0)

class TestPsychrometrics:
    """Tests for psychrometric calculations"""

    def test_psychrometric_constant(self, calc):
        """Test psychrometric constant calculation"""
        # Test at standard conditions
        gamma = calc.psychrometric_constant(20, 50, 101300)
        assert 65 < gamma < 67.1  # Typical range

        # Test pressure dependence
        gamma1 = calc.psychrometric_constant(20, 50, 101300)
        gamma2 = calc.psychrometric_constant(20, 50, 90000)
        assert gamma2 < gamma1  # Should decrease with pressure

    def test_latent_heat(self, calc):
        """Test latent heat calculation"""
        # Test at 0°C
        L = calc.latent_heat(0)
        assert np.isclose(L, 2501000, rtol=1e-3)

        # Test temperature dependence
        L1 = calc.latent_heat(0)
        L2 = calc.latent_heat(20)
        assert L2 < L1  # Should decrease with temperature


class TestAirProperties:
    """Tests for air property calculations"""

    def test_specific_heat(self, calc):
        """Test specific heat calculation"""
        # Test dry air at standard conditions
        cp = calc.specific_heat(20, 0, 101300)
        assert 1004 < cp < 1006

        # Test humidity dependence
        cp1 = calc.specific_heat(20, 0, 101300)
        cp2 = calc.specific_heat(20, 100, 101300)
        assert cp2 > cp1  # Should increase with humidity

    def test_air_density(self, calc):
        """Test air density calculation"""
        # Test at standard conditions
        rho = calc.air_density(20, 50, 101300)
        assert 1.1 < rho < 1.3  # Typical range

        # Test temperature dependence
        rho1 = calc.air_density(0, 50, 101300)
        rho2 = calc.air_density(30, 50, 101300)
        assert rho2 < rho1  # Should decrease with temperature

    def test_potential_temperature(self, calc):
        """Test potential temperature calculation"""
        # At reference pressure, should equal actual temperature
        theta = calc.potential_temperature(20, 50, 100000)
        assert np.isclose(theta, 20)

        # Test pressure dependency
        theta1 = calc.potential_temperature(20, 50, 90000)
        assert theta1 > 20  # Should be higher at lower pressure


class TestSolarCalculations:
    """Tests for solar calculations"""

    def test_solar_parameters(self, calc):
        """Test solar parameter calculations"""
        # Test at equator on equinox
        solar = calc.solar_parameters(80, 0)
        assert np.isclose(solar.max_sunshine_hours, 12, rtol=1e-2)

        # Test latitude dependence
        solar1 = calc.solar_parameters(172, 0)  # Equator
        solar2 = calc.solar_parameters(172, 45)  # Mid latitude
        assert solar2.max_sunshine_hours > solar1.max_sunshine_hours  # Summer in NH

        # Test seasonal variation
        summer = calc.solar_parameters(172, 45)  # June 21
        winter = calc.solar_parameters(355, 45)  # December 21
        assert summer.extraterrestrial_radiation > winter.extraterrestrial_radiation


class TestArrayOperations:
    """Tests for array operation handling"""

    def test_broadcasting(self, calc):
        """Test array broadcasting"""
        temps = np.array([0, 10, 20])
        rh = 50
        result = calc.actual_vapor_pressure(temps, rh)
        assert result.shape == temps.shape

    def test_mixed_dimensions(self, calc):
        """Test mixing of scalar and array inputs"""
        temps = np.array([0, 10, 20])
        rh = np.array([40, 50, 60])
        pressure = 101300
        result = calc.psychrometric_constant(temps, rh, pressure)
        assert result.shape == temps.shape


def test_value_warnings():
    """Test warning generation for borderline values"""
    calc = MeteoCalculator(MeteoConfig(raise_warnings=True))

    with pytest.warns(UserWarning):
        calc.solar_parameters(180, 70)  # Latitude > 67°

    with pytest.warns(UserWarning):
        calc.latent_heat(-50)  # Temperature out of range


def test_disable_validation():
    """Test disabling input validation"""
    calc = MeteoCalculator(MeteoConfig(validate_inputs=False))

    # Should not raise error with invalid inputs
    calc.specific_heat(np.nan, -10, -1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])