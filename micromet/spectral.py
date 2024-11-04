from scipy import signal, stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_spectra(data, sampling_freq=20, detrend=True, window='hann',
                 nfft=None, scaling='density'):
    """
    Calculate power spectra and frequency for time series data with proper variance preservation.

    Parameters:
    -----------
    data : array_like, pandas.Series
        Time series data to analyze
    sampling_freq : float, optional
        Sampling frequency in Hz, default is 20 Hz
    detrend : bool, optional
        Whether to detrend the data before analysis, default True
    window : str, optional
        Window function to use, default 'hann'
    nfft : int, optional
        Length of FFT, default None uses next power of 2
    scaling : str, optional
        Scaling of spectra, either 'density' or 'spectrum', default 'density'

    Returns:
    --------
    freqs : ndarray
        Frequencies in Hz
    Pxx : ndarray
        Power spectral density or power spectrum
    """
    # Input validation and conversion
    if isinstance(data, pd.Series):
        data = data.values
    elif not isinstance(data, (list, np.ndarray)):
        raise ValueError("Input data must be array-like or pandas Series")

    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("Input data array is empty")

    # Remove NaN values
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Remove mean and detrend if requested
    data = data - np.mean(data)
    if detrend:
        data = signal.detrend(data)

    # Set up FFT parameters
    if nfft is None:
        nfft = min(int(2 ** np.ceil(np.log2(len(data) / 8))), len(data))

    # Set segment length and overlap
    nperseg = nfft
    noverlap = nperseg // 2

    # Get window and normalize
    win = signal.get_window(window, nperseg)
    # Normalize window for proper scaling
    win_norm = np.sum(win ** 2) / nperseg
    win = win / np.sqrt(win_norm)

    try:
        freqs, Pxx = signal.welch(data, fs=sampling_freq, window=win,
                                  nperseg=nperseg, noverlap=noverlap,
                                  nfft=nfft, scaling=scaling,
                                  detrend=False)  # Already detrended if requested
    except ValueError as e:
        raise ValueError(f"Error calculating spectrum: {str(e)}")

    # Scale PSD for variance preservation
    if scaling == 'density':
        Pxx = Pxx * sampling_freq

    return freqs, Pxx

def calc_cospectra(x, y, sampling_freq=20, detrend=True, window='hann',
                   nfft=None, scaling='density'):
    """
    Calculate co-spectra between two time series with proper normalization.

    Parameters:
    -----------
    x, y : array_like
        Time series data arrays to analyze
    sampling_freq : float, optional
        Sampling frequency in Hz, default is 20 Hz
    detrend : bool, optional
        Whether to detrend the data before analysis, default True
    window : str, optional
        Window function to use, default 'hann'
    nfft : int, optional
        Length of FFT, default None uses next power of 2
    scaling : str, optional
        Either 'density' or 'spectrum', default 'density'

    Returns:
    --------
    freqs : ndarray
        Frequencies in Hz
    Cxy : ndarray
        Normalized co-spectrum between x and y
    """
    # Input validation
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("Input arrays must have same length")

    # Remove mean before processing
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Detrend if requested
    if detrend:
        x = signal.detrend(x)
        y = signal.detrend(y)

    # Set up FFT parameters
    if nfft is None:
        # Use segment length about 1/8 of total length
        nfft = min(int(2 ** np.ceil(np.log2(len(x) / 8))), len(x))

    # Set segment length and overlap
    nperseg = nfft
    noverlap = nperseg // 2

    # Get window and normalize it
    win = signal.get_window(window, nperseg)
    win_norm = np.sqrt(np.mean(win ** 2))
    win = win / win_norm

    # Initialize arrays for averaging
    n_segments = (len(x) - noverlap) // (nperseg - noverlap)
    freqs = np.fft.rfftfreq(nperseg, d=1 / sampling_freq)
    Pxy_avg = np.zeros(len(freqs), dtype=complex)

    # Process segments
    for i in range(n_segments):
        start = i * (nperseg - noverlap)
        end = start + nperseg

        # Apply window
        x_seg = win * x[start:end]
        y_seg = win * y[start:end]

        # Compute FFTs
        fx = np.fft.rfft(x_seg)
        fy = np.fft.rfft(y_seg)

        # Accumulate cross-spectrum
        Pxy_avg += fx * np.conjugate(fy)

    # Average across segments
    Pxy_avg /= n_segments

    # Extract real part (co-spectrum)
    Cxy = np.real(Pxy_avg)

    # Apply scaling
    if scaling == 'density':
        # Normalize to power spectral density
        # Factor of 2 accounts for one-sided spectrum
        Cxy *= 1.0 / (sampling_freq * nperseg)
    else:  # scaling == 'spectrum'
        # Normalize to power spectrum
        Cxy *= 1.0 / nperseg

    return freqs, Cxy

def spectral_analysis(df, variables=['Ux', 'Uy', 'Uz', 'Ts', 'pV'],
                      sampling_freq=20):
    """
    Perform spectral analysis on eddy covariance data.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing high frequency eddy covariance data
    variables : list of str
        List of variable names to analyze, default ['Ux','Uy','Uz','Ts','pV']
    sampling_freq : float
        Sampling frequency in Hz, default 20 Hz

    Returns:
    --------
    dict
        Dictionary containing:
        - 'spectra': Power spectra for each variable
        - 'cospectra': Co-spectra between vertical wind (Uz) and other variables
        - 'frequencies': Frequency arrays
        - 'normalized_freqs': Normalized frequencies (f*z/U)
        - 'peaks': Spectral peak information
    """
    # Check that all variables exist in dataframe
    for var in variables:
        if var not in df.columns:
            raise ValueError(f"Variable '{var}' not found in DataFrame")

    results = {
        'spectra': {},
        'cospectra': {},
        'frequencies': {},
        'normalized_freqs': {},
        'peaks': {}
    }

    # Calculate mean wind speed for normalization
    U = np.sqrt(df['Ux'].mean() ** 2 + df['Uy'].mean() ** 2)
    z = 3.0  # Measurement height in meters, should be parameterized

    # Calculate spectra for each variable
    for var in variables:
        try:
            freqs, Pxx = calc_spectra(df[var].values, sampling_freq=sampling_freq)
            results['spectra'][var] = Pxx
            results['frequencies'][var] = freqs
            results['normalized_freqs'][var] = freqs * z / U

            # Find spectral peaks
            peak_idx = signal.find_peaks(Pxx)[0]
            if len(peak_idx) > 0:
                results['peaks'][var] = {
                    'freq': freqs[peak_idx],
                    'power': Pxx[peak_idx]
                }
        except Exception as e:
            print(f"Warning: Failed to calculate spectra for {var}: {str(e)}")

    # Calculate cospectra with vertical wind component
    for var in variables:
        if var != 'Uz':
            try:
                freqs, Cxy = calc_cospectra(df['Uz'].values, df[var].values,
                                            sampling_freq=sampling_freq)
                results['cospectra'][f'Uz-{var}'] = Cxy
            except Exception as e:
                print(f"Warning: Failed to calculate cospectra for Uz-{var}: {str(e)}")

    # Calculate energy dissipation rate
    def calc_dissipation(freqs, Pxx, U):
        inertial_range = (freqs > 0.1) & (freqs < 10)
        if np.any(inertial_range):
            try:
                slope, _, _, _, _ = stats.linregress(
                    np.log(freqs[inertial_range]),
                    np.log(Pxx[inertial_range])
                )
                C = 0.55  # Kolmogorov constant
                eps = ((U * np.mean(Pxx[inertial_range]) *
                        freqs[inertial_range] ** (5 / 3)) / C) ** (3 / 2)
                return np.mean(eps)
            except Exception:
                return np.nan
        return np.nan

    results['dissipation_rate'] = {
        var: calc_dissipation(
            results['frequencies'][var],
            results['spectra'][var],
            U
        ) for var in variables
    }

    return results


def plot_spectra(results, normalized=True, loglog=True):
    """
    Plot power spectra and cospectra from spectral analysis results.

    Parameters:
    -----------
    results : dict
        Results dictionary from spectral_analysis()
    normalized : bool
        Whether to plot normalized frequencies, default True
    loglog : bool
        Whether to use log-log scaling, default True
    """
    import matplotlib.pyplot as plt

    # Plot power spectra
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for var in results['spectra'].keys():
        freqs = (results['normalized_freqs'][var] if normalized
                 else results['frequencies'][var])
        spec = results['spectra'][var]

        if loglog:
            ax1.loglog(freqs, spec, label=var)
        else:
            ax1.plot(freqs, spec, label=var)

    ax1.set_xlabel('Normalized Frequency (fz/U)' if normalized
                   else 'Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Power Spectra')

    # Plot cospectra
    for key, cospec in results['cospectra'].items():
        freqs = (results['normalized_freqs']['Uz'] if normalized
                 else results['frequencies']['Uz'])

        if loglog:
            ax2.loglog(freqs, np.abs(cospec), label=key)
        else:
            ax2.plot(freqs, cospec, label=key)

    ax2.set_xlabel('Normalized Frequency (fz/U)' if normalized
                   else 'Frequency (Hz)')
    ax2.set_ylabel('Co-spectral Density')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Co-spectra with vertical wind')

    plt.tight_layout()
    return fig, (ax1, ax2)


def generate_example_ec_data(duration=30, sampling_freq=20, include_noise=True, seed=None):
    """
    Generate synthetic eddy covariance data with realistic spectral characteristics.

    Parameters:
    -----------
    duration : float, optional
        Duration of the time series in minutes, default 30 minutes
    sampling_freq : float, optional
        Sampling frequency in Hz, default 20 Hz
    include_noise : bool, optional
        Whether to add random noise to the signals, default True
    seed : int, optional
        Random seed for reproducibility, default None

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic EC data with columns:
        - Ux, Uy, Uz: Wind components
        - Ts: Sonic temperature
        - pV: Water vapor density

    Notes:
    -----
    The function generates data with:
    - Mean wind ~3-5 m/s with turbulent fluctuations
    - Temperature fluctuations with realistic diurnal trend
    - Water vapor with correlation to temperature
    - Turbulent eddies at multiple scales
    - Optional random noise
    """
    if seed is not None:
        np.random.seed(seed)

    # Time array
    n_samples = int(duration * 60 * sampling_freq)
    t = np.linspace(0, duration * 60, n_samples)

    # Base frequencies for different scales of motion
    f_low = 1 / 600  # ~10 min period
    f_mid = 1 / 60  # ~1 min period
    f_high = 1 / 6  # ~6 sec period

    # Generate wind components
    # Mean wind ~4 m/s with turbulent fluctuations
    Ux_mean = 4.0
    Ux = Ux_mean + \
         0.8 * np.sin(2 * np.pi * f_low * t) + \
         0.4 * np.sin(2 * np.pi * f_mid * t) + \
         0.2 * np.sin(2 * np.pi * f_high * t)

    # Cross-wind component
    Uy = 0.5 * np.sin(2 * np.pi * f_low * t + np.pi / 4) + \
         0.3 * np.sin(2 * np.pi * f_mid * t + np.pi / 6) + \
         0.1 * np.sin(2 * np.pi * f_high * t + np.pi / 3)

    # Vertical wind
    Uz = 0.3 * np.sin(2 * np.pi * f_low * t + np.pi / 3) + \
         0.2 * np.sin(2 * np.pi * f_mid * t + np.pi / 2) + \
         0.1 * np.sin(2 * np.pi * f_high * t + 2 * np.pi / 3)

    # Temperature with diurnal trend
    Ts_mean = 25.0
    Ts = Ts_mean + \
         2.0 * np.sin(2 * np.pi * t / (duration * 60)) + \
         0.5 * np.sin(2 * np.pi * f_low * t) + \
         0.3 * np.sin(2 * np.pi * f_mid * t) + \
         0.1 * np.sin(2 * np.pi * f_high * t)

    # Water vapor with correlation to temperature
    pV_mean = 10.0
    pV = pV_mean + \
         0.8 * signal.detrend(Ts) + \
         0.4 * np.sin(2 * np.pi * f_low * t + np.pi / 6) + \
         0.2 * np.sin(2 * np.pi * f_mid * t + np.pi / 4) + \
         0.1 * np.sin(2 * np.pi * f_high * t + np.pi / 3)

    if include_noise:
        # Add random turbulent fluctuations
        Ux += np.random.normal(0, 0.2, n_samples)
        Uy += np.random.normal(0, 0.15, n_samples)
        Uz += np.random.normal(0, 0.1, n_samples)
        Ts += np.random.normal(0, 0.1, n_samples)
        pV += np.random.normal(0, 0.1, n_samples)

    # Create DataFrame with datetime index
    start_time = pd.Timestamp.now().round('30min')
    time_index = pd.date_range(start=start_time,
                               periods=n_samples,
                               freq=f'{1 / sampling_freq}s')

    df = pd.DataFrame({
        'Ux': Ux,
        'Uy': Uy,
        'Uz': Uz,
        'Ts': Ts,
        'pV': pV
    }, index=time_index)

    return df


def demo_spectral_analysis():
    """
    Demonstrate the spectral analysis functions using synthetic data.

    Returns:
    --------
    tuple
        (DataFrame of synthetic data, spectral analysis results, figure)
    """
    # Generate example data
    df = generate_example_ec_data(duration=30, sampling_freq=20, seed=42)

    # Perform spectral analysis
    results = spectral_analysis(df)

    # Create plots
    fig, axes = plot_spectra(results)

    # Print dataset properties using updated timedelta calculation
    print("\nDataset properties:")
    print("-" * 20)
    print(f"Duration: {(df.index[-1] - df.index[0]).total_seconds() / 60:.1f} minutes")

    # Calculate sampling frequency using timedelta between consecutive points
    sampling_freq = 1 / pd.Timedelta(df.index[1] - df.index[0]).total_seconds()
    print(f"Sampling frequency: {sampling_freq:.1f} Hz")

    print(f"\nMean values:")
    print(df.mean())
    print(f"\nStandard deviations:")
    print(df.std())

    return df, results, (fig, axes)


def calc_kaimal_spectrum(f, z=3.0, u_star=0.5, L=-50):
    """
    Calculate the theoretical Kaimal spectrum for w'T' cospectra.

    Parameters:
    -----------
    f : array_like
        Frequencies in Hz
    z : float
        Measurement height (m)
    u_star : float
        Friction velocity (m/s)
    L : float
        Obukhov length (m), negative for unstable conditions

    Returns:
    --------
    array_like
        Normalized cospectral density values
    """
    # Calculate normalized frequency
    zeta = z / L  # stability parameter
    phi = (1 - 16 * zeta) ** (-0.25)  # stability function
    n = f * z / (u_star * phi)

    # Kaimal formula for unstable conditions
    cospec = 14 * n / (1 + 9.6 * n) ** (2.4)

    return cospec


def calc_slope_line(f, z, U, slope=-2 / 3, anchor_point=(0.1, 0.2)):
    """
    Calculate a reference slope line for the inertial subrange.

    Parameters:
    -----------
    f : array_like
        Frequencies in Hz
    z : float
        Measurement height (m)
    U : float
        Mean wind speed (m/s)
    slope : float
        Desired slope, default -2/3
    anchor_point : tuple
        (x, y) coordinates to anchor the slope line in normalized coordinates

    Returns:
    --------
    tuple
        (x coordinates, y coordinates) for plotting
    """
    # Convert frequencies to normalized form
    n = f * z / U

    # Calculate y-intercept to pass through anchor point
    # y = mx + b in log space
    log_x0, log_y0 = np.log10(anchor_point[0]), np.log10(anchor_point[1])
    b = log_y0 - slope * log_x0

    # Calculate line values
    mask = (n >= anchor_point[0] / 5) & (n <= anchor_point[0] * 20)
    log_y = slope * np.log10(n[mask]) + b

    return n[mask], 10 ** log_y


def plot_wt_cospectra(df, sampling_freq=20, z=3.0, u_star=0.5, L=-50,
                      show_slope=True, slope=-2 / 3):
    """
    Plot w'T' cospectra with theoretical Kaimal spectrum and optional slope line.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing high frequency data with 'Uz' and 'Ts' columns
    sampling_freq : float
        Sampling frequency in Hz
    z : float
        Measurement height (m)
    u_star : float
        Friction velocity (m/s)
    L : float
        Obukhov length (m)
    show_slope : bool
        Whether to show the slope reference line
    slope : float
        Slope value to show, default -2/3

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Calculate cospectra
    freqs, Cxy = calc_cospectra(df['Uz'], df['Ts'], sampling_freq=sampling_freq)

    # Calculate mean wind speed for normalization
    U = np.sqrt(df['Ux'].mean() ** 2 + df['Uy'].mean() ** 2)

    # Normalize frequencies and cospectra
    n = freqs * z / U
    norm_cospec = freqs * Cxy / (u_star * df['Ts'].std())

    # Calculate theoretical Kaimal spectrum
    kaimal = calc_kaimal_spectrum(freqs, z, u_star, L)

    # Calculate slope reference line if requested
    if show_slope:
        slope_x, slope_y = calc_slope_line(freqs, z, U, slope=slope)

    # Create logarithmically spaced bins for block averaging
    log_bins = np.logspace(np.log10(n[1]), np.log10(n[-1]), 20)
    bin_means_x = []
    bin_means_y = []

    for i in range(len(log_bins) - 1):
        mask = (n >= log_bins[i]) & (n < log_bins[i + 1])
        if np.any(mask):
            bin_means_x.append(np.mean(n[mask]))
            bin_means_y.append(np.mean(norm_cospec[mask]))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw cospectra
    ax.semilogx(n, norm_cospec, 'lightgray', alpha=0.5, label='Raw cospectra')

    # Plot block averages
    ax.semilogx(bin_means_x, bin_means_y, 'ko', label='Block averages',
                markersize=8, markerfacecolor='white')

    # Plot Kaimal spectrum
    ax.semilogx(n, kaimal, 'k-', label='Kaimal spectrum', linewidth=2)

    # Plot slope reference line if requested
    if show_slope:
        ax.semilogx(slope_x, slope_y, 'r--',
                    label=f'{slope:.1f} slope', linewidth=2)

    # Set axis limits and labels
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(-0.1, 0.3)
    ax.set_xlabel('$n = fz/U$')
    ax.set_ylabel('$fCo_{wT}/u_*T_*$')

    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add legend
    ax.legend()

    # Add title
    ax.set_title("w'T' Cospectrum\nUnstable Conditions", pad=20)

    plt.tight_layout()
    return fig


def example_wt_cospectra(show_slope=True):
    """
    Generate example data and create w'T' cospectra plot.
    """
    # Generate example data with strong w-T correlation
    df = generate_example_ec_data(duration=30, sampling_freq=20, seed=42)

    # Add some artificial correlation between w and T
    df['Ts'] = df['Ts'] + 0.5 * df['Uz']

    # Create plot with slope line
    fig = plot_wt_cospectra(df, show_slope=show_slope)

    return df, fig


# Example usage
if __name__ == "__main__":
    # Generate and analyze example data
    df, results, (fig, axes) = demo_spectral_analysis()

    print("\nDataset properties:")
    print("-" * 20)
    print(f"Duration: {(df.index[-1] - df.index[0]).total_seconds() / 60:.1f} minutes")
    print(f"Sampling frequency: {1 / df.index.freq.delta.total_seconds():.1f} Hz")
    print(f"\nMean values:")
    print(df.mean())
    print(f"\nStandard deviations:")
    print(df.std())

    # Display plots
    import matplotlib.pyplot as plt

    plt.show()