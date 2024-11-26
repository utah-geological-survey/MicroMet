import pytest
import numpy as np
from micromet.ffp import (
    FootprintInput, CoordinateSystem, FootprintConfig, FootprintCalculator,
    CoordinateTransformer, EnhancedFootprintProcessor
)
from pyproj import CRS


# Test Data fixtures
@pytest.fixture
def valid_footprint_input():
    return FootprintInput(
        zm=10.0,
        z0=0.1,
        umean=5.0,
        h=1000.0,
        ol=-100.0,
        sigmav=1.0,
        ustar=0.5,
        wind_dir=180.0
    )


@pytest.fixture
def utm_crs():
    """Create UTM zone 32N CRS"""
    return CoordinateSystem.from_epsg(32632)  # UTM Zone 32N


@pytest.fixture
def small_calculator():
    """Create a calculator that uses small grids"""
    calculator = FootprintCalculator()
    original_calculate = calculator.calculate_footprint

    def calculate_small_footprint(*args, **kwargs):
        # Force small grid size
        if 'nx' not in kwargs:
            kwargs['nx'] = 50
        elif kwargs['nx'] > 100:
            kwargs['nx'] = 50
        return original_calculate(*args, **kwargs)

    calculator.calculate_footprint = calculate_small_footprint
    return calculator


@pytest.fixture
def footprint_config(utm_crs):
    """Create a config using UTM coordinates with smaller domain"""
    return FootprintConfig(
        origin_distance=1000.0,
        measurement_height=10.0,
        roughness_length=0.1,
        domain_size=(300000, 300200, 5000000, 5000200),  # 200m x 200m domain
        grid_resolution=20.0,  # 20m resolution
        station_coords=(300100, 5000100),  # Center of domain
        coordinate_system=utm_crs,
        working_crs=utm_crs
    )


@pytest.fixture
def processor_with_small_calc(footprint_config, small_calculator):
    """Create processor that uses the small calculator"""
    processor = EnhancedFootprintProcessor(footprint_config)
    processor.calculator = small_calculator
    return processor
@pytest.fixture
def coordinate_system():
    return CoordinateSystem.from_epsg(4326)  # WGS84

@pytest.fixture
def working_crs():
    return CoordinateSystem.from_epsg(32631)  # UTM Zone 31N

def calculate_nx_from_config(config: FootprintConfig) -> int:
    """Calculate appropriate nx value based on domain size and resolution"""
    domain_width = config.domain_size[1] - config.domain_size[0]
    return max(10, min(100, int(domain_width / config.grid_resolution)))

# FootprintInput Tests
def test_footprint_input_validation_valid(valid_footprint_input):
    """Test validation of valid FootprintInput"""
    assert valid_footprint_input.validate() is True

@pytest.mark.parametrize("field,invalid_value,expected_error", [
    ("zm", -1.0, "zm must be positive"),
    ("z0", -0.1, "z0 must be positive if provided"),
    ("h", 5.0, "h must be > 10m"),
    ("sigmav", -1.0, "sigmav must be positive"),
    ("ustar", 0.05, "ustar must be >= 0.1"),
    ("wind_dir", 400.0, "wind_dir must be between 0 and 360")
])
def test_footprint_input_validation_invalid(valid_footprint_input, field, invalid_value, expected_error):
    """Test validation fails with invalid inputs"""
    setattr(valid_footprint_input, field, invalid_value)
    with pytest.raises(ValueError, match=expected_error):
        valid_footprint_input.validate()

# CoordinateSystem Tests
def test_coordinate_system_from_epsg():
    """Test creating CoordinateSystem from EPSG code"""
    cs = CoordinateSystem.from_epsg(4326)
    assert cs.is_geographic is True
    assert cs.units.lower() == 'degree'
    assert 'geodetic' in cs.datum.lower()  # Updated assertion

def test_coordinate_system_from_proj():
    """Test creating CoordinateSystem from proj string"""
    proj_str = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
    cs = CoordinateSystem.from_proj(proj_str)
    assert cs.is_geographic is False
    assert cs.units.lower() == 'metre'
    assert 'geodetic' in cs.datum.lower()  # Updated assertion

# FootprintCalculator Tests
def test_footprint_calculator_basic(valid_footprint_input):
    """Test basic footprint calculation"""
    calc = FootprintCalculator()
    result = calc.calculate_footprint(valid_footprint_input)

    assert 'x_2d' in result
    assert 'y_2d' in result
    assert 'f_2d' in result
    assert isinstance(result['x_2d'], np.ndarray)
    assert isinstance(result['y_2d'], np.ndarray)
    assert isinstance(result['f_2d'], np.ndarray)
    assert result['f_2d'].shape == result['x_2d'].shape

def test_footprint_calculator_grid_size(valid_footprint_input):
    """Test footprint calculation with different grid sizes"""
    calc = FootprintCalculator()
    nx = 500
    result = calc.calculate_footprint(valid_footprint_input, nx=nx)

    assert result['x_2d'].shape == (nx, nx)
    assert result['y_2d'].shape == (nx, nx)
    assert result['f_2d'].shape == (nx, nx)

def test_footprint_calculator_climatology(valid_footprint_input):
    """Test footprint climatology calculation"""
    calc = FootprintCalculator()
    input_series = [valid_footprint_input] * 3

    result = calc.calculate_footprint_climatology(input_series)

    assert 'fclim_2d' in result
    assert 'n' in result
    assert result['n'] == 3
    assert np.all(result['fclim_2d'] >= 0)

# CoordinateTransformer Tests
def test_coordinate_transformer():
    """Test coordinate transformation between projected coordinate systems"""
    source_crs = CoordinateSystem.from_epsg(32631)  # UTM Zone 31N
    target_crs = CoordinateSystem.from_epsg(32632)  # UTM Zone 32N

    transformer = CoordinateTransformer(source_crs, target_crs)

    x = np.array([500000.0])  # Valid UTM coordinate
    y = np.array([5000000.0])  # Valid UTM coordinate

    x_trans, y_trans = transformer.transform_coords(x, y)
    x_back, y_back = transformer.transform_coords(x_trans, y_trans, direction='inverse')

    np.testing.assert_array_almost_equal(x, x_back, decimal=3)
    np.testing.assert_array_almost_equal(y, y_back, decimal=3)

# EnhancedFootprintProcessor Tests

def test_enhanced_footprint_processor(processor_with_small_calc, valid_footprint_input):
    """Test enhanced footprint processor with coordinate transformations"""
    # Calculate footprint
    result = processor_with_small_calc.calculate_georeferenced_footprint(valid_footprint_input)

    # Validate results
    assert 'x_2d' in result
    assert 'y_2d' in result
    assert 'f_2d' in result

    # Check array shapes are reasonable
    assert result['x_2d'].shape[0] <= 100
    assert result['x_2d'].shape[1] <= 100
    assert result['x_2d'].shape == result['y_2d'].shape
    assert result['f_2d'].shape == result['x_2d'].shape

    # Check for NaN values
    assert not np.any(np.isnan(result['x_2d']))
    assert not np.any(np.isnan(result['y_2d']))
    assert not np.any(np.isnan(result['f_2d']))


def test_utm_zone_calculation():
    """Test UTM zone calculation"""
    test_cases = [
        ((0, 0), 32631),  # 0° longitude -> zone 31N
        ((6, 0), 32632),  # 6° longitude -> zone 32N
        ((12, 0), 32633),  # 12° longitude -> zone 33N
        ((0, -10), 32731),  # Southern hemisphere
    ]

    config = FootprintConfig(
        origin_distance=1000.0,
        measurement_height=10.0,
        roughness_length=0.1,
        domain_size=(-1.0, 1.0, -1.0, 1.0),
        grid_resolution=0.1,
        station_coords=(0.0, 0.0),
        coordinate_system=CoordinateSystem.from_epsg(4326)
    )

    processor = EnhancedFootprintProcessor(config)

    for (lon, lat), expected_epsg in test_cases:
        calculated_epsg = processor._get_utm_zone(lon, lat)
        assert calculated_epsg == expected_epsg, \
            f"UTM zone mismatch for ({lon}, {lat}): expected {expected_epsg}, got {calculated_epsg}"


def test_footprint_calculator_basic(valid_footprint_input, small_calculator):
    """Test basic footprint calculation with small grid"""
    result = small_calculator.calculate_footprint(
        valid_footprint_input,
        domain=(-100, 100, -100, 100)
    )

    assert 'x_2d' in result
    assert 'y_2d' in result
    assert 'f_2d' in result
    assert isinstance(result['x_2d'], np.ndarray)
    assert isinstance(result['y_2d'], np.ndarray)
    assert isinstance(result['f_2d'], np.ndarray)
    assert result['f_2d'].shape[0] <= 100
    assert result['f_2d'].shape[1] <= 100


def test_full_footprint_workflow(processor_with_small_calc, valid_footprint_input):
    """Test complete footprint calculation workflow with memory-efficient settings"""
    # Calculate footprint
    result = processor_with_small_calc.calculate_georeferenced_footprint(valid_footprint_input)

    # Basic validation
    assert 'x_2d' in result
    assert 'y_2d' in result
    assert 'f_2d' in result

    # Check shapes are reasonable
    assert result['x_2d'].shape[0] <= 100
    assert result['x_2d'].shape[1] <= 100

    # Check coordinate ranges
    assert np.min(result['x_2d']) >= processor_with_small_calc.config.domain_size[0]
    assert np.max(result['x_2d']) <= processor_with_small_calc.config.domain_size[1]
    assert np.min(result['y_2d']) >= processor_with_small_calc.config.domain_size[2]
    assert np.max(result['y_2d']) <= processor_with_small_calc.config.domain_size[3]

    # Check for NaN values
    assert not np.any(np.isnan(result['x_2d']))
    assert not np.any(np.isnan(result['y_2d']))
    assert not np.any(np.isnan(result['f_2d']))

    # Check footprint properties
    assert np.all(result['f_2d'] >= 0)  # Non-negative values


def test_coordinate_transformation_integrity(processor_with_small_calc):
    """Test coordinate transformations with smaller domain"""
    # Test points within domain (using fewer points)
    test_points_x = np.linspace(
        processor_with_small_calc.config.domain_size[0],
        processor_with_small_calc.config.domain_size[1],
        5
    )
    test_points_y = np.linspace(
        processor_with_small_calc.config.domain_size[2],
        processor_with_small_calc.config.domain_size[3],
        5
    )

    X, Y = np.meshgrid(test_points_x, test_points_y)

    # Transform coordinates
    x_work, y_work = processor_with_small_calc.transform_to_working(X.flatten(), Y.flatten())
    x_orig, y_orig = processor_with_small_calc.transform_from_working(x_work, y_work)

    # Check roundtrip accuracy
    np.testing.assert_array_almost_equal(X.flatten(), x_orig, decimal=3)
    np.testing.assert_array_almost_equal(Y.flatten(), y_orig, decimal=3)

if __name__ == '__main__':
    pytest.main([__file__])
