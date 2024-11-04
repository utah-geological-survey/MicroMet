import unittest
import pytest
import numpy as np
from pyproj import CRS


from micromet.ffp import (
    FootprintInput,
    CoordinateSystem,
    FootprintCalculator,
    FootprintConfig,
    CoordinateTransformer
)


# Test FootprintInput validation
def test_footprint_input_validation():
    # Valid input
    valid_input = FootprintInput(
        zm=10.0,
        z0=0.1,
        umean=2.0,
        h=1000.0,
        ol=100.0,
        sigmav=0.5,
        ustar=0.3,
        wind_dir=180.0
    )
    assert valid_input.validate() is True

    # Invalid zm
    with pytest.raises(ValueError):
        invalid_input = FootprintInput(
            zm=-1.0,
            z0=0.1,
            umean=2.0,
            h=1000.0,
            ol=100.0,
            sigmav=0.5,
            ustar=0.3,
            wind_dir=180.0
        )
        invalid_input.validate()

    # Invalid h
    with pytest.raises(ValueError):
        invalid_input = FootprintInput(
            zm=10.0,
            z0=0.1,
            umean=2.0,
            h=5.0,  # Too small
            ol=100.0,
            sigmav=0.5,
            ustar=0.3,
            wind_dir=180.0
        )
        invalid_input.validate()


# Test CoordinateSystem creation
def test_coordinate_system():
    # Test EPSG creation
    cs = CoordinateSystem.from_epsg(4326)
    assert cs.is_geographic is True
    assert cs.units == "degree"

    # Test proj string creation
    proj_str = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs"
    cs = CoordinateSystem.from_proj(proj_str)
    assert cs.is_geographic is False
    assert cs.units == "metre"


# Test FootprintCalculator basic functionality
def test_footprint_calculator():
    calc = FootprintCalculator()

    # Test basic footprint calculation
    inputs = FootprintInput(
        zm=10.0,
        z0=0.1,
        umean=None,
        h=1000.0,
        ol=100.0,
        sigmav=0.5,
        ustar=0.3,
        wind_dir=None
    )

    result = calc.calculate_footprint(inputs)

    assert 'x_2d' in result
    assert 'y_2d' in result
    assert 'f_2d' in result
    assert isinstance(result['x_2d'], np.ndarray)
    assert isinstance(result['y_2d'], np.ndarray)
    assert isinstance(result['f_2d'], np.ndarray)


# Test CoordinateTransformer
def test_coordinate_transformer():
    source_crs = CoordinateSystem.from_epsg(4326)  # WGS84
    target_crs = CoordinateSystem.from_epsg(32632)  # UTM 32N

    transformer = CoordinateTransformer(source_crs, target_crs)

    # Test point transformation
    lon, lat = [12.0], [45.0]
    x, y = transformer.transform_coords(lon, lat)

    assert len(x) == 1
    assert len(y) == 1
    assert x[0] > 0  # Should be in meters
    assert y[0] > 0  # Should be in meters


def test_footprint_config():
    config = FootprintConfig(
        origin_distance=1000.0,
        measurement_height=10.0,
        roughness_length=0.1,
        domain_size=(-1000.0, 1000.0, -1000.0, 1000.0),
        grid_resolution=10.0,
        station_coords=(45.0, 12.0),
        coordinate_system=CoordinateSystem.from_epsg(4326)
    )

    assert config.origin_distance == 1000.0
    assert config.measurement_height == 10.0
    assert config.roughness_length == 0.1
    assert len(config.domain_size) == 4
    assert config.grid_resolution == 10.0
    assert len(config.station_coords) == 2


# Test smoothing function
def test_footprint_smoothing():
    calc = FootprintCalculator()
    test_array = np.ones((10, 10))
    smoothed = calc._smooth_footprint(test_array)

    assert smoothed.shape == test_array.shape
    assert np.all(np.isfinite(smoothed))


def test_calc_xstar_methods():
    """Test xstar calculation methods"""
    calc = FootprintCalculator()

    # Test input data
    inputs = FootprintInput(
        zm=10.0,
        z0=0.1,
        umean=2.0,
        h=1000.0,
        ol=-100.0,  # Unstable conditions
        sigmav=0.5,
        ustar=0.3,
        wind_dir=None
    )

    # Create test grid
    x = np.linspace(-100, 100, 10)
    y = np.linspace(-100, 100, 10)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(xx, yy)

    # Test z0 method
    xstar_z0 = calc._calc_xstar_z0(inputs, rho, theta)
    assert xstar_z0.shape == rho.shape
    assert np.all(np.isfinite(xstar_z0))

    # Test umean method
    xstar_umean = calc._calc_xstar_umean(inputs, rho, theta)
    assert xstar_umean.shape == rho.shape
    assert np.all(np.isfinite(xstar_umean))


def test_calc_footprint_values():
    """Test footprint value calculations"""
    calc = FootprintCalculator()

    # Test input data
    inputs = FootprintInput(
        zm=10.0,
        z0=0.1,
        umean=2.0,
        h=1000.0,
        ol=-100.0,
        sigmav=0.5,
        ustar=0.3,
        wind_dir=None
    )

    # Create test grid - using positive x values to ensure valid footprint region
    x = np.linspace(0, 100, 10)  # Only positive x values
    y = np.linspace(-50, 50, 10)  # Symmetric around y=0
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    # Calculate xstar
    xstar = calc._calc_xstar_z0(inputs, rho, theta)

    # Test footprint value calculation
    f_2d = calc._calc_footprint_values(inputs, xstar, rho, theta)

    assert f_2d.shape == rho.shape
    assert np.all(np.isfinite(f_2d))  # All values should be finite
    assert np.all(f_2d >= 0)  # All values should be non-negative

    # Check that we have some non-zero values
    assert np.sum(f_2d > 0) > 0

    # Check maximum location is reasonable (should be downstream of measurement)
    max_idx = np.unravel_index(np.argmax(f_2d), f_2d.shape)
    assert xx[max_idx] > 0  # Maximum should be downstream


def test_different_stability_conditions():
    """Test calculations under different stability conditions"""
    calc = FootprintCalculator()

    # Create test grid
    x = np.linspace(-100, 100, 10)
    y = np.linspace(-100, 100, 10)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(xx, yy)

    # Test cases: unstable, neutral, and stable conditions
    test_cases = [
        {'ol': -100.0, 'case': 'unstable'},
        {'ol': 1E6, 'case': 'neutral'},
        {'ol': 100.0, 'case': 'stable'}
    ]

    for case in test_cases:
        inputs = FootprintInput(
            zm=10.0,
            z0=0.1,
            umean=2.0,
            h=1000.0,
            ol=case['ol'],
            sigmav=0.5,
            ustar=0.3,
            wind_dir=None
        )

        xstar = calc._calc_xstar_z0(inputs, rho, theta)
        f_2d = calc._calc_footprint_values(inputs, xstar, rho, theta)

        assert np.all(np.isfinite(f_2d)), f"Failed for {case['case']} conditions"
        assert np.all(f_2d >= 0), f"Negative values found for {case['case']} conditions"