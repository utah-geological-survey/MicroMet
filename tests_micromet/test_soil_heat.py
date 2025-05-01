import pytest
import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append("../src")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from micromet.soil_heat import (
    temperature_gradient,
    soil_heat_flux,
    volumetric_heat_capacity,
    thermal_conductivity,
    diurnal_amplitude,
    diurnal_peak_lag,
    thermal_diffusivity_amplitude,
    thermal_diffusivity_lag,
    thermal_diffusivity_logrithmic,
    calculate_thermal_diffusivity_for_pair,
    calculate_thermal_properties_for_all_pairs,
)


# Test data setup
@pytest.fixture
def sample_temperature_data():
    """Create sample temperature data with datetime index"""
    dates = pd.date_range(start="2023-01-01", periods=48, freq="H")
    temp_shallow = 20 + 5 * np.sin(np.linspace(0, 4 * np.pi, 48))  # Daily cycle
    temp_deep = 20 + 2 * np.sin(
        np.linspace(0, 4 * np.pi, 48) + np.pi / 4
    )  # Damped and lagged
    df = pd.DataFrame(
        {
            "datetime": dates,
            "temp_5cm": temp_shallow,
            "temp_20cm": temp_deep,
            "swc_5cm": [0.25] * 48,
            "swc_20cm": [0.30] * 48,
        }
    )
    df.set_index("datetime", inplace=True)
    return df


@pytest.fixture
def depth_mapping():
    return {"temp_5cm": 0.05, "temp_20cm": 0.20}


# Test cases
def test_temperature_gradient():
    # Test with simple values
    assert temperature_gradient(20, 22, 0.1, 0.3) == -10.0
    assert temperature_gradient(22, 20, 0.1, 0.3) == 10.0
    assert temperature_gradient(15, 15, 0.1, 0.3) == 0.0

    # Test with different depths
    assert temperature_gradient(20, 22, 0.05, 0.15) == -20.0


def test_soil_heat_flux():
    # Test with known values
    assert pytest.approx(soil_heat_flux(20, 22, 0.1, 0.3, 1.0), 0.001) == 10.0
    assert pytest.approx(soil_heat_flux(22, 20, 0.1, 0.3, 0.5), 0.001) == -5.0

    # Test with zero gradient
    assert soil_heat_flux(20, 20, 0.1, 0.3, 1.0) == 0.0


def test_volumetric_heat_capacity():
    # Test with known values
    dry_soil = volumetric_heat_capacity(0.0)
    assert pytest.approx(dry_soil, 0.001) == 1942.0

    water = volumetric_heat_capacity(1.0)
    assert pytest.approx(water, 0.001) == 4186.0

    mixed = volumetric_heat_capacity(0.25)
    assert pytest.approx(mixed, 0.001) == (0.75 * 1942 + 0.25 * 4186)

    # Test with edge cases
    with pytest.raises(ValueError):
        volumetric_heat_capacity(-0.1)
    with pytest.raises(ValueError):
        volumetric_heat_capacity(1.1)


def test_thermal_conductivity():
    # Test with known values
    alpha = 1e-6  # m²/s
    theta_v = 0.25

    # Expected Cv = (1-0.25)*1942 + 0.25*4186 = 2503 kJ/(m³·°C)
    expected_k = 1e-6 * 2503 * 1000  # Convert kJ to J
    assert pytest.approx(thermal_conductivity(alpha, theta_v), 0.001) == expected_k

    # Test with zero moisture
    assert thermal_conductivity(alpha, 0.0) == alpha * 1942 * 1000


def test_diurnal_amplitude(sample_temperature_data):
    # Test with sample data
    amplitude = diurnal_amplitude(sample_temperature_data["temp_5cm"])

    # Should have one value per day (2 days in sample data)
    assert len(amplitude) == 2

    # Amplitude should be approximately 10 (max-min for sine wave with amplitude 5)
    assert pytest.approx(amplitude.iloc[0], 0.1) == 10.0


def test_diurnal_peak_lag(sample_temperature_data):
    # Test with sample data
    lag = diurnal_peak_lag(
        sample_temperature_data["temp_5cm"], sample_temperature_data["temp_20cm"]
    )

    # Should have one value per day
    assert len(lag) == 2

    # The lag should be approximately 6 hours (pi/4 phase shift in daily cycle)
    assert pytest.approx(lag.iloc[0], 0.1) == -6.0


def test_thermal_diffusivity_amplitude():
    # Test with known values
    A1 = 10.0  # Amplitude at shallow depth
    A2 = 5.0  # Amplitude at deeper depth
    z1 = 0.05  # 5 cm
    z2 = 0.20  # 20 cm

    alpha = thermal_diffusivity_amplitude(A1, A2, z1, z2)

    # Expected value: π*(0.15)^2 / (86400 * (ln(2))^2) ≈ 1.07e-6
    expected = (np.pi * (0.15) ** 2) / (86400 * (np.log(2)) ** 2)
    assert pytest.approx(alpha, 1e-8) == expected

    # Test with equal amplitudes (should return inf or handle appropriately)
    with pytest.raises(ValueError):
        thermal_diffusivity_amplitude(10.0, 10.0, z1, z2)


def test_thermal_diffusivity_lag():
    # Test with known values
    delta_t = 6 * 3600  # 6 hours in seconds
    z1 = 0.05  # 5 cm
    z2 = 0.20  # 20 cm

    alpha = thermal_diffusivity_lag(delta_t, z1, z2)

    # Expected value: (86400/(4π)) * (0.15)^2 / (21600)^2 ≈ 1.65e-7
    expected = (86400 / (4 * np.pi)) * (0.15) ** 2 / (21600) ** 2
    assert pytest.approx(alpha, 1e-8) == expected

    # Test with zero lag (should raise error)
    with pytest.raises(ZeroDivisionError):
        thermal_diffusivity_lag(0, z1, z2)


def test_thermal_diffusivity_logarithmic():
    # Test with known values
    # Using temperatures that would result in a specific ratio
    alpha = thermal_diffusivity_logrithmic(
        25,
        20,
        15,
        20,  # z1 temps (complete cycle)
        22,
        20,
        18,
        20,  # z2 temps (damped cycle)
        0.05,
        0.20,
    )

    # The exact expected value depends on the ratio calculation
    assert alpha > 0  # Should be positive

    # Test with insufficient data
    with pytest.raises(IndexError):
        thermal_diffusivity_logrithmic(25, 20, 15, 20, 22, 20, 18, 20, 0.05, 0.20)


def test_calculate_thermal_diffusivity_for_pair(sample_temperature_data, depth_mapping):
    # Test with sample data
    results = calculate_thermal_diffusivity_for_pair(
        sample_temperature_data, "temp_5cm", "temp_20cm", 0.05, 0.20
    )

    # Should return a dictionary with three methods
    assert set(results.keys()) == {"alpha_log", "alpha_amp", "alpha_lag"}

    # All values should be positive
    assert all(v > 0 for v in results.values() if v is not None)


def test_calculate_thermal_properties_for_all_pairs(
    sample_temperature_data, depth_mapping
):
    # Rename columns to match depth_mapping
    sample_temperature_data = sample_temperature_data.rename(
        columns={
            "temp_5cm": "ts_3_1_1",
            "temp_20cm": "ts_3_3_1",
            "swc_5cm": "swc_3_1_1",
            "swc_20cm": "swc_3_3_1",
        }
    )

    depth_mapping = {"ts_3_1_1": 0.05, "ts_3_3_1": 0.20}

    results = calculate_thermal_properties_for_all_pairs(
        sample_temperature_data, depth_mapping
    )

    # Should return a DataFrame with results
    assert isinstance(results, pd.DataFrame)
    assert not results.empty

    # Should contain expected columns
    expected_cols = ["alpha_amp", "G_amp", "k_amp", "theta_v"]
    assert all(col in results.columns for col in expected_cols)


def test_edge_cases():
    # Test temperature gradient with zero depth difference
    with pytest.raises(ZeroDivisionError):
        temperature_gradient(20, 22, 0.1, 0.1)

    # Test soil heat flux with invalid conductivity
    with pytest.raises(TypeError):
        soil_heat_flux(20, 22, 0.1, 0.3, "invalid")

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(KeyError):
        calculate_thermal_diffusivity_for_pair(empty_df, "col1", "col2", 0.1, 0.2)


if __name__ == "__main__":
    pytest.main()  # Run the tests in this file
