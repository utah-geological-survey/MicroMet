import pytest
import numpy as np
import pandas as pd
from scipy import signal

from micromet.spectral import (
    calc_spectra, 
    calc_cospectra,
    spectral_analysis,
    calc_kaimal_spectrum,
    calc_slope_line,
    generate_example_ec_data,
    plot_wt_cospectra
)

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Create synthetic time series
    t = np.linspace(0, 30*60, 36000)  # 30 minutes at 20 Hz
    f1, f2 = 0.1, 1.0  # Two test frequencies
    
    # Create test signals
    signal1 = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
    signal2 = np.cos(2*np.pi*f1*t) + 0.3*np.cos(2*np.pi*f2*t)
    
    return signal1, signal2, t

@pytest.fixture
def sample_ec_data():
    """Generate sample eddy covariance data."""
    return generate_example_ec_data(duration=30, sampling_freq=20, seed=42)

def test_calc_spectra_basic(sample_data):
    """Test basic functionality of calc_spectra."""
    signal1, _, _ = sample_data
    
    freqs, Pxx = calc_spectra(signal1, sampling_freq=20)
    
    # Test output shapes and types
    assert isinstance(freqs, np.ndarray)
    assert isinstance(Pxx, np.ndarray)
    assert len(freqs) == len(Pxx)
    
    # Test frequency range
    assert freqs[0] >= 0
    assert freqs[-1] <= 10  # Nyquist frequency
    
    # Test for positive power values
    assert np.all(Pxx >= 0)

def test_calc_spectra_peaks(sample_data):
    """Test if calc_spectra correctly identifies spectral peaks."""
    signal1, _, _ = sample_data
    
    freqs, Pxx = calc_spectra(signal1, sampling_freq=20)
    
    # Find peaks in spectrum
    peak_idx = signal.find_peaks(Pxx)[0]
    peak_freqs = freqs[peak_idx]
    
    # Test if peaks are near expected frequencies (0.1 and 1.0 Hz)
    assert any(abs(peak_freqs - 0.1) < 0.05)
    assert any(abs(peak_freqs - 1.0) < 0.05)

def test_calc_cospectra_basic(sample_data):
    """Test basic functionality of calc_cospectra."""
    signal1, signal2, _ = sample_data
    
    freqs, Cxy = calc_cospectra(signal1, signal2, sampling_freq=20)
    
    # Test output shapes and types
    assert isinstance(freqs, np.ndarray)
    assert isinstance(Cxy, np.ndarray)
    assert len(freqs) == len(Cxy)
    
    # Test frequency range
    assert freqs[0] >= 0
    assert freqs[-1] <= 10

def test_spectral_analysis_output(sample_ec_data):
    """Test the output structure of spectral_analysis."""
    results = spectral_analysis(sample_ec_data)
    
    # Check required keys
    required_keys = ['spectra', 'cospectra', 'frequencies', 
                    'normalized_freqs', 'peaks']
    for key in required_keys:
        assert key in results
    
    # Check variables
    variables = ['Ux', 'Uy', 'Uz', 'Ts', 'pV']
    for var in variables:
        assert var in results['spectra']
    
    # Check cospectra
    for var in variables:
        if var != 'Uz':
            assert f'Uz-{var}' in results['cospectra']

def test_calc_kaimal_spectrum():
    """Test calculation of theoretical Kaimal spectrum."""
    f = np.logspace(-3, 1, 100)
    spectrum = calc_kaimal_spectrum(f, z=3.0, u_star=0.5, L=-50)
    
    # Test output shape and type
    assert isinstance(spectrum, np.ndarray)
    assert len(spectrum) == len(f)
    
    # Test for positive values
    assert np.all(spectrum >= 0)
    
    # Test peak location (should be around n ≈ 0.1)
    peak_idx = np.argmax(spectrum)
    n = f[peak_idx] * 3.0 / 0.5  # n = fz/U
    assert 0.05 < n < 0.2

def test_calc_slope_line():
    """Test calculation of reference slope line."""
    f = np.logspace(-3, 1, 100)
    x, y = calc_slope_line(f, z=3.0, U=5.0, slope=-2/3)
    
    # Test output shapes
    assert len(x) == len(y)
    
    # Test slope
    log_x = np.log10(x[1:])
    log_y = np.log10(y[1:])
    slope = np.diff(log_y) / np.diff(log_x)
    assert np.allclose(slope.mean(), -2/3, rtol=0.1)

def test_plot_wt_cospectra(sample_ec_data):
    """Test plot generation."""
    fig = plot_wt_cospectra(sample_ec_data)
    
    # Test figure properties
    assert fig is not None
    assert len(fig.axes) > 0
    
    # Test axis labels
    ax = fig.axes[0]
    assert 'n = fz/U' in ax.get_xlabel()
    assert 'fCo' in ax.get_ylabel()

def test_generate_example_ec_data():
    """Test synthetic data generation."""
    df = generate_example_ec_data(duration=30, sampling_freq=20)
    
    # Test DataFrame structure
    required_cols = ['Ux', 'Uy', 'Uz', 'Ts', 'pV']
    for col in required_cols:
        assert col in df.columns
    
    # Test data properties
    assert len(df) == 30 * 60 * 20  # duration * seconds * freq
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Test reproducibility with seed
    df1 = generate_example_ec_data(seed=42)
    df2 = generate_example_ec_data(seed=42)
    assert np.allclose(df1['Ux'].values, df2['Ux'].values)


def test_error_handling():
    """Test error handling in various functions."""
    # Test invalid input to calc_spectra
    with pytest.raises(ValueError, match="Input data array is empty"):
        calc_spectra(np.array([]))

    with pytest.raises(ValueError, match="Input data must be array-like"):
        calc_spectra(None)

    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        calc_spectra(np.array([1, 2, 3]), sampling_freq=0)

    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        calc_spectra(np.array([1, 2, 3]), sampling_freq=-1)

    # Test invalid scaling parameter
    with pytest.raises(ValueError, match="Scaling must be either 'density' or 'spectrum'"):
        calc_spectra(np.array([1, 2, 3]), scaling='invalid')

    # Test all-NaN input
    with pytest.raises(ValueError, match="No valid data points after removing NaN values"):
        calc_spectra(np.array([np.nan, np.nan, np.nan]))

    # Test invalid nfft
    with pytest.raises(ValueError, match="nfft must be a positive integer"):
        calc_spectra(np.array([1, 2, 3]), nfft=0)

    # Test mismatched arrays in calc_cospectra
    with pytest.raises(ValueError, match="Input arrays must have same length"):
        calc_cospectra(np.array([1, 2, 3]), np.array([1, 2]))

    with pytest.raises(ValueError, match="Inputs must be array-like"):
        calc_cospectra(None, np.array([1, 2, 3]))

    # Test empty arrays in calc_cospectra
    with pytest.raises(ValueError, match="Input arrays are empty"):
        calc_cospectra(np.array([]), np.array([]))

    # Test all-NaN input in calc_cospectra
    with pytest.raises(ValueError, match="No valid data points after removing NaN values"):
        calc_cospectra(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))


def test_edge_cases():
    """Test behavior with edge cases."""
    # Test single value
    freqs, Pxx = calc_spectra(np.array([1.0]))
    assert len(freqs) > 0
    assert len(Pxx) > 0

    # Test very small values
    freqs, Pxx = calc_spectra(np.array([1e-10, 1e-10]))
    assert np.all(np.isfinite(Pxx))

    # Test very large values
    freqs, Pxx = calc_spectra(np.array([1e10, 1e10]))
    assert np.all(np.isfinite(Pxx))

    # Test mix of positive and negative values
    freqs, Pxx = calc_spectra(np.array([-1, 1, -1, 1]))
    assert np.all(np.isfinite(Pxx))


def test_input_types():
    """Test different input types."""
    # Test list input
    freqs1, Pxx1 = calc_spectra([1, 2, 3, 4])

    # Test numpy array input
    freqs2, Pxx2 = calc_spectra(np.array([1, 2, 3, 4]))

    # Results should be identical
    assert np.allclose(freqs1, freqs2)
    assert np.allclose(Pxx1, Pxx2)

    # Test float32 input
    freqs3, Pxx3 = calc_spectra(np.array([1, 2, 3, 4], dtype=np.float32))
    assert np.allclose(Pxx1, Pxx3, rtol=1e-6)

    # Test integer input
    freqs4, Pxx4 = calc_spectra(np.array([1, 2, 3, 4], dtype=np.int32))
    assert np.allclose(Pxx1, Pxx4, rtol=1e-6)


def test_spectral_properties(sample_ec_data):
    """
    Test physical properties of spectra including variance preservation.
    """
    results = spectral_analysis(sample_ec_data)

    for var in ['Ux', 'Uy', 'Uz']:
        # Calculate variance in time domain
        data = sample_ec_data[var].values
        data = data - np.mean(data)  # Remove mean
        var_time = np.var(data)

        # Get spectrum
        freqs = results['frequencies'][var]
        Pxx = results['spectra'][var]

        # Calculate variance from PSD using proper scaling
        df = freqs[1] - freqs[0]  # Frequency spacing
        var_freq = np.sum(Pxx) * df

        # Compare variances
        rel_diff = np.abs(var_time - var_freq) / var_time
        assert rel_diff < 0.3, \
            f"Variance mismatch for {var}: time={var_time:.3f}, freq={var_freq:.3f}, rel_diff={rel_diff:.3f}"


def test_spectrum_white_noise():
    """
    Test spectral calculation with white noise signal.
    """
    np.random.seed(42)

    # Generate white noise with known variance
    n_samples = 72000
    variance = 1.0
    x = np.random.normal(0, np.sqrt(variance), n_samples)

    # Calculate spectrum
    freqs, Pxx = calc_spectra(x, sampling_freq=20)

    # Calculate variance from spectrum
    df = freqs[1] - freqs[0]
    var_freq = np.sum(Pxx) * df

    # Compare with actual variance
    rel_diff = np.abs(variance - var_freq) / variance
    assert rel_diff < 0.2, \
        f"White noise variance error: expected {variance}, got {var_freq}, rel_diff={rel_diff:.3f}"


def test_spectrum_sine_wave():
    """
    Test spectral calculation with sine wave of known amplitude.
    """
    # Create sine wave with known amplitude
    t = np.linspace(0, 3600, 72000)  # 1 hour at 20 Hz
    f0 = 0.1  # 0.1 Hz
    A = 2.0  # Amplitude
    x = A * np.sin(2 * np.pi * f0 * t)

    # Theoretical variance = A^2/2
    var_theory = A ** 2 / 2

    # Calculate spectrum
    freqs, Pxx = calc_spectra(x, sampling_freq=20)

    # Calculate variance from spectrum
    df = freqs[1] - freqs[0]
    var_freq = np.sum(Pxx) * df

    # Compare variances
    rel_diff = np.abs(var_theory - var_freq) / var_theory
    assert rel_diff < 0.2, \
        f"Sine wave variance error: expected {var_theory}, got {var_freq}, rel_diff={rel_diff:.3f}"


def test_spectral_scaling():
    """
    Test different spectral scaling options.
    """
    # Generate test signal
    t = np.linspace(0, 3600, 72000)
    x = np.sin(2 * np.pi * 0.1 * t)

    # Calculate spectra with different scaling
    freqs1, Pxx1 = calc_spectra(x, scaling='density')
    freqs2, Pxx2 = calc_spectra(x, scaling='spectrum')

    # Verify that freqs are identical
    assert np.allclose(freqs1, freqs2)

    # Calculate variance from both
    df = freqs1[1] - freqs1[0]
    var1 = np.sum(Pxx1) * df
    var2 = np.sum(Pxx2) * df / 20  # Divide by sampling frequency for spectrum scaling

    # Variances should be close
    assert np.abs(var1 - var2) / var1 < 0.1, \
        f"Scaling inconsistency: {var1} vs {var2}"

def test_cospectra_noise():
    """
    Test cospectra behavior with uncorrelated noise.
    """
    np.random.seed(42)
    n_samples = 72000

    # Generate uncorrelated white noise
    x = np.random.normal(0, 1, n_samples)
    y = np.random.normal(0, 1, n_samples)

    # Calculate cospectra
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    # Calculate integrated covariance
    df = freqs[1] - freqs[0]
    cov_spec = 2 * np.sum(cospec) * df

    # For uncorrelated signals, should be close to zero
    assert np.abs(cov_spec) < 0.1, \
        f"Cospectrum integral of uncorrelated noise too large: {np.abs(cov_spec):.3f}"


def test_cospectra_consistency():
    """
    Test consistency between time and frequency domain calculations using
    synthetic signals with known properties.
    """
    # Generate test signal with multiple frequencies
    t = np.linspace(0, 3600, 72000)  # 1 hour at 20 Hz
    f1, f2 = 0.1, 1.0  # Two test frequencies
    a1, a2 = 1.0, 0.5  # Amplitudes

    # Create signals with known relationship
    x = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    y = 0.8 * a1 * np.sin(2 * np.pi * f1 * t) + 0.4 * a2 * np.sin(2 * np.pi * f2 * t)

    # Theoretical covariance and peak calculations
    # For sine waves:
    # cov(A1*sin(ω1t), B1*sin(ω1t)) = A1*B1/2
    # First component: (1.0 * 0.8)/2 = 0.4
    # Second component: (0.5 * 0.4)/2 = 0.1
    # Total covariance = 0.4 + 0.1 = 0.5
    theoretical_cov = 0.5

    # Calculate time domain covariance
    cov_time = np.mean((x - np.mean(x)) * (y - np.mean(y)))

    # Verify time domain calculation
    assert np.abs(cov_time - theoretical_cov) < 0.01, \
        f"Time domain calculation error: {np.abs(cov_time - theoretical_cov):.3f}"

    # Calculate frequency domain covariance
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    # Integrate cospectrum
    df = freqs[1] - freqs[0]
    cov_freq = 2 * np.sum(cospec) * df

    # Compare with theoretical value
    rel_diff_theory = np.abs(cov_freq - theoretical_cov) / np.abs(theoretical_cov)
    assert rel_diff_theory < 0.3, \
        f"Theory vs frequency domain error: {rel_diff_theory:.3f}"

    # Compare time and frequency domain
    rel_diff = np.abs(cov_freq - cov_time) / np.abs(cov_time)
    assert rel_diff < 0.3, f"Time vs frequency domain error: {rel_diff:.3f}"

    def get_peak_value(freq, freqs, cospec, window=0.02):
        """Get peak value in spectrum near specified frequency"""
        mask = (freqs >= freq - window) & (freqs <= freq + window)
        if not np.any(mask):
            return None
        idx = np.argmax(np.abs(cospec[mask]))
        idx_global = np.where(mask)[0][idx]
        return freqs[idx_global], np.abs(cospec[idx_global])

    # Find peaks near input frequencies
    peak1_res = get_peak_value(f1, freqs, cospec)
    peak2_res = get_peak_value(f2, freqs, cospec)

    # Verify peaks were found
    assert peak1_res is not None, f"No peak found near {f1} Hz"
    assert peak2_res is not None, f"No peak found near {f2} Hz"

    # Unpack peak results
    f1_found, peak1 = peak1_res
    f2_found, peak2 = peak2_res

    # Check peak frequencies
    assert np.abs(f1_found - f1) < 0.02, \
        f"First peak at wrong frequency: expected {f1}, got {f1_found}"
    assert np.abs(f2_found - f2) < 0.02, \
        f"Second peak at wrong frequency: expected {f2}, got {f2_found}"

    # Calculate amplitude ratios
    # The co-spectral peak amplitudes should be proportional to the
    # product of input amplitudes at each frequency
    amp_ratio_1 = a1 * 0.8  # amplitude product at f1
    amp_ratio_2 = a2 * 0.4  # amplitude product at f2
    expected_ratio = amp_ratio_1 / amp_ratio_2

    measured_ratio = peak1 / peak2

    # Calculate relative error in logarithmic space to handle ratios better
    log_ratio_error = np.abs(np.log(measured_ratio) - np.log(expected_ratio))
    assert log_ratio_error < np.log(2), \
        f"Peak ratio error too large: expected ratio {expected_ratio:.3f}, got {measured_ratio:.3f}"


def test_cospectra_simple():
    """
    Test cospectra calculation with simple single-frequency signals.
    """
    # Create simple test signals
    t = np.linspace(0, 3600, 72000)  # 1 hour at 20 Hz
    f0 = 0.1  # Single frequency

    # Create signals with known amplitude relationship
    x = np.sin(2 * np.pi * f0 * t)
    y = 0.5 * np.sin(2 * np.pi * f0 * t)  # Half amplitude

    # Calculate cospectra
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    # Find peak
    peak_idx = np.argmax(np.abs(cospec))
    peak_freq = freqs[peak_idx]
    peak_value = np.abs(cospec[peak_idx])

    # Check peak frequency
    assert np.abs(peak_freq - f0) < 0.02, \
        f"Peak at wrong frequency: expected {f0}, got {peak_freq}"

    # For simple sine waves, verify covariance
    theoretical_cov = 0.5 * 0.5  # product of amplitudes / 2

    # Calculate covariance from spectrum
    df = freqs[1] - freqs[0]
    cov = 2 * np.sum(cospec) * df

    rel_error = np.abs(cov - theoretical_cov) / theoretical_cov
    assert rel_error < 0.2, \
        f"Covariance error too large: expected {theoretical_cov}, got {cov}"


def test_phase_relationships():
    """
    Test cospectra behavior with different phase relationships.
    """
    t = np.linspace(0, 3600, 72000)
    f0 = 0.1

    # In-phase signals
    x1 = np.sin(2 * np.pi * f0 * t)
    y1 = 0.5 * np.sin(2 * np.pi * f0 * t)

    # Out-of-phase signals
    x2 = np.sin(2 * np.pi * f0 * t)
    y2 = 0.5 * np.sin(2 * np.pi * f0 * t + np.pi)

    # Calculate cospectra
    _, cospec1 = calc_cospectra(x1, y1, sampling_freq=20)
    _, cospec2 = calc_cospectra(x2, y2, sampling_freq=20)

    # Out-of-phase signals should have opposite sign
    assert np.sum(cospec1) * np.sum(cospec2) < 0, \
        "Out-of-phase signals should have opposite-sign cospectra"


def test_spectral_peaks():
    """
    Test detection and measurement of spectral peaks using simple signals.
    """
    # Create test signal with two clear peaks
    t = np.linspace(0, 3600, 72000)
    f1, f2 = 0.1, 1.0
    x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    y = 0.8 * np.sin(2 * np.pi * f1 * t) + 0.2 * np.sin(2 * np.pi * f2 * t)

    # Calculate cospectra
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    def find_peaks_in_range(freqs, cospec, freq_range, n_peaks=1):
        """Find highest peaks in frequency range"""
        start_freq, end_freq = freq_range
        mask = (freqs >= start_freq) & (freqs <= end_freq)
        if not np.any(mask):
            return []

        # Find peaks in masked region
        peaks, properties = signal.find_peaks(np.abs(cospec[mask]))
        if len(peaks) == 0:
            return []

        # Sort by amplitude
        peak_amplitudes = np.abs(cospec[mask][peaks])
        sorted_idx = np.argsort(peak_amplitudes)[::-1]  # Descending order
        return peaks[sorted_idx[:n_peaks]]

    # Find peaks in low and high frequency ranges
    low_peaks = find_peaks_in_range(freqs, cospec, (0.05, 0.15))
    high_peaks = find_peaks_in_range(freqs, cospec, (0.9, 1.1))

    assert len(low_peaks) > 0, "No peak found near 0.1 Hz"
    assert len(high_peaks) > 0, "No peak found near 1.0 Hz"

    # Calculate peak frequencies
    mask_low = (freqs >= 0.05) & (freqs <= 0.15)
    mask_high = (freqs >= 0.9) & (freqs <= 1.1)

    low_freq = freqs[mask_low][low_peaks[0]]
    high_freq = freqs[mask_high][high_peaks[0]]

    # Check peak frequencies
    assert np.abs(low_freq - f1) < 0.05, \
        f"Low frequency peak at wrong frequency: {low_freq:.3f} Hz"
    assert np.abs(high_freq - f2) < 0.05, \
        f"High frequency peak at wrong frequency: {high_freq:.3f} Hz"

    # Check peak amplitudes
    low_amp = np.abs(cospec[mask_low][low_peaks[0]])
    high_amp = np.abs(cospec[mask_high][high_peaks[0]])

    # Ratio of peak amplitudes should reflect signal amplitudes
    measured_ratio = low_amp / high_amp
    expected_ratio = (0.8 * 1.0) / (0.2 * 0.5)  # (0.8*1.0)/(0.2*0.5) = 8

    rel_ratio_error = np.abs(measured_ratio - expected_ratio) / expected_ratio
    assert rel_ratio_error < 0.5, \
        f"Peak amplitude ratio error: {rel_ratio_error:.3f}"

def test_cospectra_basic_sine():
    """
    Test cospectra calculation with simple sine waves.
    """
    # Create simple test signal
    t = np.linspace(0, 3600, 72000)
    f0 = 0.1  # Single frequency component

    # In-phase signals with different amplitudes
    x = np.sin(2 * np.pi * f0 * t)
    y = 0.5 * np.sin(2 * np.pi * f0 * t)

    # Theoretical covariance = 0.25
    theoretical_cov = 0.25

    # Calculate time domain covariance
    cov_time = np.mean((x - np.mean(x)) * (y - np.mean(y)))

    # Calculate frequency domain covariance
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)
    df = freqs[1] - freqs[0]
    cov_freq = 2 * np.sum(cospec) * df

    # Verify results
    assert np.abs(cov_time - theoretical_cov) < 0.01, \
        f"Time domain error: {np.abs(cov_time - theoretical_cov):.3f}"

    assert np.abs(cov_freq - theoretical_cov) < 0.01, \
        f"Frequency domain error: {np.abs(cov_freq - theoretical_cov):.3f}"


def test_cospectra_normalization():
    """
    Test proper normalization of cospectra using signals with unit amplitude.
    """
    # Create two sine waves with unit amplitude and known phase difference
    t = np.linspace(0, 100, 2000)  # 100 seconds at 20 Hz
    f0 = 0.5  # 0.5 Hz

    x = np.sin(2 * np.pi * f0 * t)
    y = np.sin(2 * np.pi * f0 * t + np.pi / 4)  # 45 degree phase shift

    # Calculate cospectra
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    # For unit amplitude sine waves with 45° phase shift:
    # The coherence should be 1 at f0
    # The cospectrum should peak at f0 with magnitude cos(π/4) = 1/√2
    peak_idx = np.argmax(np.abs(cospec))
    peak_freq = freqs[peak_idx]
    peak_value = cospec[peak_idx]

    # Check peak frequency
    assert np.abs(peak_freq - f0) < 0.1, f"Peak frequency error: {peak_freq - f0:.3f}"

    # Normalize peak value considering Welch's method scaling
    df = np.mean(np.diff(freqs))
    scale_factor = 2 * df  # Factor of 2 for one-sided spectrum
    norm_peak = peak_value * scale_factor

    # Expected value is cos(π/4) = 1/√2 ≈ 0.707
    expected_value = 1 / np.sqrt(2)
    assert np.abs(norm_peak - expected_value) < 0.2, \
        f"Peak value error: {norm_peak - expected_value:.3f}"


def test_cospectra_with_real_data(sample_ec_data):
    """
    Test cospectra properties with real eddy covariance data.
    """
    results = spectral_analysis(sample_ec_data)

    # Get vertical wind and temperature data
    uz = sample_ec_data['Uz'].values
    ts = sample_ec_data['Ts'].values

    # Remove means (but not trends for covariance comparison)
    uz = uz - np.mean(uz)
    ts = ts - np.mean(ts)

    # Calculate direct covariance
    cov_time = np.mean(uz * ts)

    # Calculate cospectra directly for comparison
    freqs, cospec = calc_cospectra(uz, ts, sampling_freq=20)

    # Calculate covariance from cospectrum
    df = freqs[1] - freqs[0]  # Frequency spacing
    cov_freq = np.sum(cospec) * df

    # Calculate relative difference
    rel_diff = np.abs(cov_freq - cov_time) / np.abs(cov_time)
    assert rel_diff < 0.3, f"Relative difference too large: {rel_diff:.3f}"

    # Check physical properties
    # 1. Max frequency check
    nyquist = 10.0  # Nyquist frequency for 20 Hz sampling
    assert np.all(freqs <= nyquist), "Frequencies exceed Nyquist"

    # 2. Check frequency spacing
    freq_spacing = np.diff(freqs)
    assert np.allclose(freq_spacing, freq_spacing[0], rtol=1e-10), \
        "Uneven frequency spacing"

    # 3. Check inertial subrange scaling
    inertial_mask = (freqs >= 0.1) & (freqs <= 1.0)
    if np.sum(inertial_mask) > 10:  # Ensure enough points for fit
        log_freqs = np.log10(freqs[inertial_mask])
        log_cospec = np.log10(np.abs(cospec[inertial_mask]))

        # Fit slope in log-log space
        slope, _ = np.polyfit(log_freqs, log_cospec, 1)

        # For w'T' cospectra, expect -7/3 slope in inertial subrange
        # Allow for some deviation due to measurement noise and finite sampling
        expected_slope = -7 / 3
        assert -4 < slope < -1, \
            f"Inertial subrange slope outside physical range: {slope:.2f}"


def test_cospectra_synthetic():
    """
    Test cospectra calculations with synthetic data having known properties.
    """
    # Create synthetic time series with known covariance
    t = np.linspace(0, 3600, 72000)  # 1 hour at 20 Hz
    f1 = 0.1  # Low frequency component

    # Create two correlated signals
    x = np.sin(2 * np.pi * f1 * t)
    y = 0.5 * np.sin(2 * np.pi * f1 * t)  # Same frequency, different amplitude

    # Known covariance = 0.25 for these signals
    expected_cov = 0.25

    # Calculate cospectra
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    # Calculate covariance from cospectrum
    df = freqs[1] - freqs[0]
    cov_from_spec = np.sum(cospec) * df

    # Compare with expected value
    rel_error = np.abs(cov_from_spec - expected_cov) / expected_cov
    assert rel_error < 0.2, f"Relative error too large: {rel_error:.3f}"


def test_cospectra_properties():
    """
    Test physical properties of cospectra using synthetic data with known properties.
    """
    # Generate synthetic data with known covariance
    t = np.linspace(0, 3600, 72000)  # 1 hour at 20 Hz
    f1 = 0.1  # 0.1 Hz oscillation

    # Create simple correlated signals
    x = np.sin(2 * np.pi * f1 * t)
    y = 0.5 * np.sin(2 * np.pi * f1 * t)  # Half amplitude

    # Known covariance = 0.25
    theoretical_cov = 0.25

    # Calculate covariance in time domain
    cov_time = np.cov(x, y)[0, 1]

    # Calculate cospectra
    freqs, cospec = calc_cospectra(x, y, sampling_freq=20)

    # Calculate covariance from cospectrum
    df = freqs[1] - freqs[0]
    cov_freq = np.sum(cospec) * df

    # Compare covariances
    rel_diff = np.abs(cov_time - cov_freq) / np.abs(cov_time)
    assert rel_diff < 0.3, f"Covariance mismatch: {rel_diff:.3f}"

    # Test spectral properties
    peak_idx = signal.find_peaks(np.abs(cospec))[0]
    peak_freqs = freqs[peak_idx]

    # Should have peak at f1
    assert any(np.abs(peak_freqs - f1) < 0.05), "Missing expected spectral peak"


def test_cospectra_noise_rejection():
    """
    Test that cospectra properly handle uncorrelated noise.
    """
    np.random.seed(42)
    n_samples = 72000  # 1 hour at 20 Hz

    # Generate uncorrelated white noise
    noise1 = np.random.normal(0, 1, n_samples)
    noise2 = np.random.normal(0, 1, n_samples)

    # Calculate cospectra
    freqs, cospec = calc_cospectra(noise1, noise2, sampling_freq=20)

    # For uncorrelated noise:
    # 1. Mean of cospectrum should be near zero
    assert np.abs(np.mean(cospec)) < 0.1, \
        f"Mean cospectrum too large: {np.abs(np.mean(cospec)):.3f}"

    # 2. RMS should be small
    rms = np.sqrt(np.mean(cospec ** 2))
    assert rms < 0.1, f"RMS of noise cospectrum too large: {rms:.3f}"

    # 3. Integrated covariance should be small
    df = freqs[1] - freqs[0]
    cov = np.sum(cospec) * df
    assert np.abs(cov) < 0.1, f"Integrated covariance too large: {np.abs(cov):.3f}"


def test_cospectra_scaling():
    """
    Test different scaling options in cospectra calculation.
    """
    # Generate test signals
    t = np.linspace(0, 3600, 72000)
    f1 = 0.1
    x = np.sin(2 * np.pi * f1 * t)
    y = 0.5 * np.sin(2 * np.pi * f1 * t)

    # Calculate with both scaling options
    freqs1, cospec1 = calc_cospectra(x, y, scaling='density')
    freqs2, cospec2 = calc_cospectra(x, y, scaling='spectrum')

    # Verify frequency arrays are identical
    assert np.allclose(freqs1, freqs2)

    # Verify scaling relationship
    # spectrum = density * sampling_freq
    sampling_freq = 20
    ratio = cospec2 / cospec1
    assert np.allclose(ratio[ratio != 0], sampling_freq, rtol=0.1)



def test_inertial_subrange_scaling(sample_ec_data):
    """Test scaling in the inertial subrange."""
    results = spectral_analysis(sample_ec_data)

    freqs = results['frequencies']['Ux']
    Pxx = results['spectra']['Ux']

    # Select inertial subrange (typically 0.1 to 1 Hz)
    mask = (freqs >= 0.1) & (freqs <= 1.0)

    if np.any(mask):
        # Calculate slope in log-log space
        log_freqs = np.log10(freqs[mask])
        log_power = np.log10(Pxx[mask])

        # Fit line to log-log plot
        coeffs = np.polyfit(log_freqs, log_power, 1)
        slope = coeffs[0]

        # In inertial subrange:
        # velocity spectra should follow -5/3 law
        # temperature spectra should follow -5/3 law
        expected_slope = -5 / 3

        # Allow for some deviation due to finite sampling and windowing
        assert abs(slope - expected_slope) < 0.5, f"Incorrect inertial subrange slope: {slope:.2f}"


def test_normalization():
    """Test proper normalization of spectra."""
    # Generate white noise with known variance
    np.random.seed(42)
    x = np.random.normal(0, 1, 10000)

    # Calculate spectrum
    freqs, Pxx = calc_spectra(x, sampling_freq=20, scaling='density')

    # Calculate variance from PSD
    df = np.mean(np.diff(freqs))
    var_from_psd = np.sum(Pxx) * df

    # Compare with actual variance
    actual_var = np.var(x)

    # Should be close to 1 (within 20%)
    assert abs(var_from_psd - actual_var) / actual_var < 0.2


def test_nonstationarity_detection(sample_ec_data):
    """
    Test detection of nonstationary conditions through spectral analysis.
    """
    # Create stationary and nonstationary data
    nonstat_data = sample_ec_data.copy()
    t = np.linspace(0, 1, len(nonstat_data))
    nonstat_data['Ux'] += 5 * t  # Add linear trend

    # Calculate spectra
    freqs_stat, Pxx_stat = calc_spectra(sample_ec_data['Ux'])
    freqs_nonstat, Pxx_nonstat = calc_spectra(nonstat_data['Ux'])

    # Nonstationary data should have more low-frequency power
    # Calculate power in low frequency band (f < 0.01 Hz)
    low_freq_mask = freqs_stat < 0.01

    low_freq_power_stat = np.sum(Pxx_stat[low_freq_mask])
    low_freq_power_nonstat = np.sum(Pxx_nonstat[low_freq_mask])

    # Nonstationary data should have significantly more low-frequency power
    assert low_freq_power_nonstat > 2 * low_freq_power_stat, \
        "Nonstationary condition not detected in spectrum"

    # Also check slope of low-frequency spectrum
    if np.any(low_freq_mask):
        # Calculate spectral slope in log-log space
        log_freq_stat = np.log10(freqs_stat[low_freq_mask])
        log_power_stat = np.log10(Pxx_stat[low_freq_mask])
        log_power_nonstat = np.log10(Pxx_nonstat[low_freq_mask])

        # Fit slopes
        slope_stat = np.polyfit(log_freq_stat, log_power_stat, 1)[0]
        slope_nonstat = np.polyfit(log_freq_stat, log_power_nonstat, 1)[0]

        # Nonstationary data should have steeper low-frequency slope
        assert slope_nonstat < slope_stat, \
            f"Expected steeper spectral slope for nonstationary data: {slope_nonstat:.2f} vs {slope_stat:.2f}"


def test_input_types():
    """
    Test that spectral calculation works with different input types.
    """
    # Create test data
    t = np.linspace(0, 100, 2000)
    x = np.sin(2 * np.pi * 0.1 * t)

    # Test numpy array input
    freqs1, Pxx1 = calc_spectra(x)

    # Test list input
    freqs2, Pxx2 = calc_spectra(x.tolist())

    # Test pandas Series input
    freqs3, Pxx3 = calc_spectra(pd.Series(x))

    # Results should be identical
    assert np.allclose(Pxx1, Pxx2)
    assert np.allclose(Pxx1, Pxx3)
    assert np.allclose(freqs1, freqs2)
    assert np.allclose(freqs1, freqs3)

    # Test invalid inputs
    with pytest.raises(ValueError):
        calc_spectra(None)

    with pytest.raises(ValueError):
        calc_spectra([])

    with pytest.raises(ValueError):
        calc_spectra(np.array([]))

    # Test NaN handling
    x_with_nans = x.copy()
    x_with_nans[10:20] = np.nan
    freqs4, Pxx4 = calc_spectra(x_with_nans)

    # Should still get valid output
    assert not np.any(np.isnan(Pxx4))
    assert len(freqs4) > 0

if __name__ == '__main__':
    pytest.main([__file__])
