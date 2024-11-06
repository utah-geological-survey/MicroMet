"""
Spectral and cospectral analysis module for eddy covariance data

This module provides classes and functions for analyzing spectra and cospectra 
of eddy covariance measurements, including computation of spectral corrections
and theoretical models.

Main features:
- Computation of spectra and cospectra 
- Theoretical spectral models (Kaimal, Moncrieff)
- Transfer function calculations
- Spectral corrections
"""

import numpy as np
from scipy import fft
from typing import Optional, Tuple, Dict
"""
Enhanced spectral and cospectral analysis module for eddy covariance data.
Adds advanced functionality for comprehensive spectral corrections and analysis.
"""

import numpy as np
from scipy import fft, optimize, stats
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


@dataclass
class SpectralParameters:
    """Parameters for spectral calculations"""
    f_peak: float  # Peak frequency
    broadness: float  # Spectral broadness parameter
    slope: float  # Inertial subrange slope
    R2: float  # Goodness of fit


class StabilityClass(Enum):
    """Atmospheric stability classification"""
    VERY_UNSTABLE = -2
    UNSTABLE = -1
    NEUTRAL = 0
    STABLE = 1
    VERY_STABLE = 2



class SpectralAnalysis:
    """Main class for spectral analysis of eddy covariance data"""
    
    def __init__(self, sampling_freq: float):
        """
        Initialize spectral analysis
        
        Args:
            sampling_freq: Sampling frequency in Hz
        """
        self.fs = sampling_freq
        
    def calculate_spectra(self, data: np.ndarray, detrend: bool = True,
                         window: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectra of input data
        
        Args:
            data: Input time series
            detrend: Apply linear detrending if True
            window: Apply Hanning window if True
            
        Returns:
            Tuple of:
            - Frequencies
            - Power spectra
        """
        if detrend:
            data = self._detrend(data)
            
        if window:
            window = np.hanning(len(data))
            data = data * window
            
        # Calculate FFT
        fft_vals = fft.fft(data)
        fft_freq = fft.fftfreq(len(data), 1/self.fs)
        
        # Get positive frequencies
        pos_freq_idx = fft_freq >= 0
        freqs = fft_freq[pos_freq_idx]
        
        # Calculate power spectra
        power_spectra = np.abs(fft_vals[pos_freq_idx])**2
        
        return freqs, power_spectra
    
    def calculate_cospectra(self, data1: np.ndarray, data2: np.ndarray,
                           detrend: bool = True, window: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cospectra between two time series
        
        Args:
            data1: First input time series  
            data2: Second input time series
            detrend: Apply linear detrending if True
            window: Apply Hanning window if True
            
        Returns:
            Tuple of:
            - Frequencies 
            - Cospectra
        """
        if detrend:
            data1 = self._detrend(data1)
            data2 = self._detrend(data2)
            
        if window:
            window = np.hanning(len(data1))
            data1 = data1 * window
            data2 = data2 * window
            
        # Calculate FFTs
        fft1 = fft.fft(data1)
        fft2 = fft.fft(data2)
        fft_freq = fft.fftfreq(len(data1), 1/self.fs)
        
        # Get positive frequencies
        pos_freq_idx = fft_freq >= 0
        freqs = fft_freq[pos_freq_idx]
        
        # Calculate cospectra as real part of cross-spectra
        cospectra = np.real(fft1.conj() * fft2)[pos_freq_idx]
        
        return freqs, cospectra
    
    @staticmethod
    def _detrend(data: np.ndarray) -> np.ndarray:
        """Remove linear trend from data"""
        x = np.arange(len(data))
        fit = np.polyfit(x, data, 1)
        trend = np.poly1d(fit)(x)
        return data - trend


class SpectralModels:
    """Enhanced spectral models with additional formulations"""

    @staticmethod
    def moncrieff_cospectrum(freq: np.ndarray, u_star: float, z: float,
                             L: float, var_type: str = 'scalar') -> np.ndarray:
        """
        Calculate Moncrieff et al (1997) cospectrum

        Args:
            freq: Frequencies (Hz)
            u_star: Friction velocity (m/s)
            z: Measurement height (m)
            L: Obukhov length (m)
            var_type: Type of variable ('scalar' or 'momentum')

        Returns:
            Normalized cospectrum values
        """
        # Calculate normalized frequency
        f = freq * z / u_star  # Normalized frequency
        cospec = np.zeros_like(f)  # Initialize output array

        # Stability parameter
        zeta = z / L

        if zeta > 0:  # Stable conditions
            if var_type == 'scalar':
                # Scalar fluxes (T, CO2, H2O)
                Ac = 0.284 * ((1.0 + 6.4 * zeta) ** 0.75)
                Bc = 2.34 * (Ac ** (-1.1))
                cospec = f / (f * (Ac + Bc * f ** 2.1))
            else:
                # Reynolds stress
                Au = 0.124 * ((1.0 + 7.9 * zeta) ** 0.75)
                Bu = 2.34 * (Au ** (-1.1))
                cospec = f / (f * (Au + Bu * f ** 2.1))

        else:  # Unstable conditions
            # Handle scalar fluxes
            if var_type == 'scalar':
                mask_low = f < 0.54
                mask_high = ~mask_low

                # Low frequency range
                cospec[mask_low] = 12.92 * f[mask_low] / \
                                   (1.0 + 26.7 * f[mask_low]) ** 1.375

                # High frequency range
                cospec[mask_high] = 4.378 * f[mask_high] / \
                                    (1.0 + 3.8 * f[mask_high]) ** 2.4

            else:  # Reynolds stress
                mask_low = f < 0.24
                mask_high = ~mask_low

                # Low frequency range
                cospec[mask_low] = 20.78 * f[mask_low] / \
                                   (1.0 + 31.0 * f[mask_low]) ** 1.575

                # High frequency range
                cospec[mask_high] = 12.66 * f[mask_high] / \
                                    (1.0 + 9.6 * f[mask_high]) ** 2.4

        return cospec

    def kolmogorov_spectrum(self, freq: np.ndarray, epsilon: float,
                            z: float) -> np.ndarray:
        """
        Calculate Kolmogorov spectrum in inertial subrange

        Args:
            freq: Frequencies
            epsilon: Energy dissipation rate (m²/s³)
            z: Measurement height (m)

        Returns:
            Spectral values following -5/3 law
        """
        # Kolmogorov constant
        alpha = 0.55

        # Calculate wavenumber
        k = 2 * np.pi * freq

        # Return spectrum following k^(-5/3) law
        return alpha * epsilon ** (2 / 3) * k ** (-5 / 3)

    def kaimal_spectrum(self, freq: np.ndarray, u_star: float,
                        z: float, L: float) -> np.ndarray:
        """
        Calculate Kaimal spectrum

        Args:
            freq: Frequencies
            u_star: Friction velocity
            z: Height
            L: Obukhov length

        Returns:
            Spectral values
        """
        # Normalized frequency
        f = freq * z / u_star

        # Calculate spectrum based on stability
        if L > 0:  # Stable
            return 0.164 * f / (1 + 5.3 * f ** (5 / 3))
        else:  # Unstable
            return 0.164 * f / (1 + 0.164 * (f ** (5 / 3)))

    def smooth_transition_spectrum(self, freq: np.ndarray, u_star: float,
                                   z: float, L: float,
                                   stability: str) -> np.ndarray:
        """
        Calculate spectrum with smooth transition between stability regimes

        Args:
            freq: Frequencies
            u_star: Friction velocity
            z: Height
            L: Obukhov length
            stability: Stability class

        Returns:
            Spectral values
        """
        # Get base unstable/stable spectra
        f = freq * z / u_star
        s_unstable = self.kaimal_spectrum(freq, u_star, z, -abs(L))
        s_stable = self.kaimal_spectrum(freq, u_star, z, abs(L))

        # Calculate transition weight based on stability
        if stability == 'neutral':
            weight = 0.5
        elif stability in ['unstable', 'very_unstable']:
            weight = 1.0
        else:
            weight = 0.0

        # Return weighted combination
        return weight * s_unstable + (1 - weight) * s_stable



class TransferFunctions:
    """Class for calculating system transfer functions"""
    
    @staticmethod
    def path_averaging(freq: np.ndarray, path_length: float, 
                      wind_speed: float) -> np.ndarray:
        """
        Calculate transfer function for path averaging
        
        Args:
            freq: Frequencies in Hz
            path_length: Path length of sensor (m)
            wind_speed: Mean wind speed (m/s)
            
        Returns:
            Transfer function values
        """
        k = 2 * np.pi * freq / wind_speed  # Wave number
        return np.sinc(k * path_length / 2)
    
    @staticmethod
    def sensor_separation(freq: np.ndarray, separation: float,
                         wind_speed: float) -> np.ndarray:
        """
        Calculate transfer function for sensor separation
        
        Args:
            freq: Frequencies in Hz
            separation: Sensor separation distance (m)
            wind_speed: Mean wind speed (m/s)
            
        Returns:
            Transfer function values
        """
        k = 2 * np.pi * freq / wind_speed
        return np.exp(-k * separation)
    
    @staticmethod
    def time_response(freq: np.ndarray, tau: float) -> np.ndarray:
        """
        Calculate transfer function for sensor time response
        
        Args:
            freq: Frequencies in Hz
            tau: Time constant (s)
            
        Returns:
            Transfer function values
        """
        return 1 / np.sqrt(1 + (2 * np.pi * freq * tau)**2)


class SpectralCorrections:
    """Class for calculating spectral corrections"""
    
    def __init__(self, sampling_freq: float):
        """
        Initialize spectral corrections
        
        Args:
            sampling_freq: Sampling frequency in Hz
        """
        self.fs = sampling_freq
        self.tf = TransferFunctions()
        
    def calculate_correction(self, measured_flux: float,
                           freqs: np.ndarray,
                           cospectra: np.ndarray,
                           transfer_func: np.ndarray) -> float:
        """
        Calculate spectral correction factor
        
        Args:
            measured_flux: Uncorrected flux
            freqs: Frequencies
            cospectra: Measured or model cospectra
            transfer_func: Combined transfer function
            
        Returns:
            Correction factor
        """
        # Integrate cospectra
        theoretical = np.trapz(cospectra, freqs)
        attenuated = np.trapz(cospectra * transfer_func, freqs)
        
        # Calculate correction factor
        return theoretical / attenuated




class AdvancedSpectralAnalysis(SpectralAnalysis):
    """Enhanced spectral analysis with advanced features"""

    def fit_spectrum(self, freqs: np.ndarray, spectrum: np.ndarray,
                     f_min: float = 0.001, f_max: float = 10.0) -> SpectralParameters:
        """
        Fit observed spectrum to theoretical model following Fratini et al.

        Args:
            freqs: Frequency array
            spectrum: Spectral values
            f_min: Minimum frequency for fitting
            f_max: Maximum frequency for fitting

        Returns:
            SpectralParameters with fitted values
        """
        # Mask frequencies within fitting range
        mask = (freqs >= f_min) & (freqs <= f_max)
        f_fit = freqs[mask]
        s_fit = spectrum[mask]

        def spectral_model(f: np.ndarray, f_peak: float,
                           broadness: float, slope: float) -> np.ndarray:
            """General model for spectra/cospectra"""
            return (f / f_peak) / (1 + (f / f_peak) ** (broadness * slope))

        # Initial parameter guesses
        p0 = [0.1, 1.0, 5 / 3]  # peak freq, broadness, slope

        # Fit model using non-linear least squares
        popt, pcov = optimize.curve_fit(spectral_model, f_fit, s_fit, p0=p0, maxfev = 8000)

        # Calculate R-squared
        residuals = s_fit - spectral_model(f_fit, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((s_fit - np.mean(s_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return SpectralParameters(
            f_peak=popt[0],
            broadness=popt[1],
            slope=popt[2],
            R2=r2
        )

    def calculate_ogive(self, freqs: np.ndarray, cospec: np.ndarray) -> np.ndarray:
        """
        Calculate ogive (cumulative co-spectrum)

        Args:
            freqs: Frequencies
            cospec: Cospectral values

        Returns:
            Ogive values
        """
        # Sort in ascending frequency order
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        cospec = cospec[sort_idx]

        # Cumulative integration from high to low frequencies
        return np.cumsum(cospec[::-1])[::-1]

    def estimate_noise_floor(self, spectrum: np.ndarray,
                             percentile: float = 95) -> float:
        """
        Estimate noise floor in spectrum using high frequency behavior

        Args:
            spectrum: Input spectrum
            percentile: Percentile to use for noise estimation

        Returns:
            Estimated noise floor value
        """
        # Use upper portion of spectrum
        n_points = len(spectrum)
        high_freq_idx = int(0.8 * n_points)
        high_freq_spectrum = spectrum[high_freq_idx:]

        # Estimate noise as high percentile
        return np.percentile(high_freq_spectrum, percentile)


class EnhancedSpectralModels(SpectralModels):
    """Enhanced spectral models with additional formulations"""

    def kolmogorov_spectrum(self, freq: np.ndarray, epsilon: float,
                            z: float) -> np.ndarray:
        """
        Calculate Kolmogorov spectrum in inertial subrange

        Args:
            freq: Frequencies
            epsilon: Energy dissipation rate (m²/s³)
            z: Measurement height (m)

        Returns:
            Spectral values following -5/3 law
        """
        # Kolmogorov constant
        alpha = 0.55

        # Calculate wavenumber
        k = 2 * np.pi * freq

        # Return spectrum following k^(-5/3) law
        return alpha * epsilon ** (2 / 3) * k ** (-5 / 3)

    def smooth_transition_spectrum(self, freq: np.ndarray, u_star: float,
                                   z: float, L: float, stability: StabilityClass) -> np.ndarray:
        """
        Calculate spectrum with smooth transition between stability regimes

        Args:
            freq: Frequencies
            u_star: Friction velocity
            z: Height
            L: Obukhov length
            stability: Stability class

        Returns:
            Spectral values
        """
        # Get base unstable/stable spectra
        f = freq * z / u_star
        s_unstable = self.kaimal_spectrum(freq, u_star, z, -abs(L))
        s_stable = self.kaimal_spectrum(freq, u_star, z, abs(L))

        # Calculate transition weight based on stability
        if stability == StabilityClass.NEUTRAL:
            weight = 0.5
        elif stability in [StabilityClass.UNSTABLE, StabilityClass.VERY_UNSTABLE]:
            weight = 1.0
        else:
            weight = 0.0

        # Return weighted combination
        return weight * s_unstable + (1 - weight) * s_stable


class EnhancedTransferFunctions(TransferFunctions):
    """Enhanced transfer functions with additional formulations"""

    def tube_attenuation(self, freq: np.ndarray, tube_length: float,
                         flow_rate: float, tube_diameter: float) -> np.ndarray:
        """
        Calculate transfer function for tube attenuation

        Args:
            freq: Frequencies
            tube_length: Tube length (m)
            flow_rate: Flow rate (L/min)
            tube_diameter: Tube inner diameter (m)

        Returns:
            Transfer function values
        """
        # Calculate tube parameters
        flow_rate_m3s = flow_rate / 60000  # Convert L/min to m³/s
        tube_area = np.pi * (tube_diameter / 2) ** 2
        velocity = flow_rate_m3s / tube_area

        # Calculate attenuation coefficient
        omega = 2 * np.pi * freq
        tau = tube_length / velocity

        return np.exp(-omega * tau)

    def block_averaging(self, freq: np.ndarray,
                        averaging_time: float) -> np.ndarray:
        """
        Calculate transfer function for block averaging

        Args:
            freq: Frequencies
            averaging_time: Averaging period (s)

        Returns:
            Transfer function values
        """
        x = np.pi * freq * averaging_time
        return np.sinc(x)

    def combined_transfer(self, freq: np.ndarray,
                          transfer_funcs: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple transfer functions

        Args:
            freq: Frequencies
            transfer_funcs: List of transfer functions to combine

        Returns:
            Combined transfer function
        """
        # Multiply all transfer functions
        combined = np.ones_like(freq)
        for tf in transfer_funcs:
            combined *= tf
        return combined


class SpectralQC:
    """Quality control for spectral calculations"""

    def __init__(self, min_freq: float = 0.0001, max_freq: float = 50.0):
        """
        Initialize spectral QC

        Args:
            min_freq: Minimum valid frequency
            max_freq: Maximum valid frequency
        """
        self.min_freq = min_freq
        self.max_freq = max_freq

    def check_resolution(self, freqs: np.ndarray) -> bool:
        """Check if frequency resolution is adequate"""
        freq_res = np.mean(np.diff(freqs))
        return freq_res <= 0.1 * self.min_freq

    def check_slope(self, freqs: np.ndarray, spectrum: np.ndarray,
                    freq_range: Tuple[float, float] = (0.1, 1.0),
                    expected_slope: float = -5 / 3,
                    tolerance: float = 0.2) -> bool:
        """
        Check if spectrum follows expected slope in inertial subrange

        Args:
            freqs: Frequencies
            spectrum: Spectral values
            freq_range: Frequency range for slope calculation
            expected_slope: Expected spectral slope
            tolerance: Allowable deviation from expected slope

        Returns:
            True if slope is within tolerance
        """
        # Get data in frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        f_fit = freqs[mask]
        s_fit = spectrum[mask]

        # Fit log-log slope
        slope, _, r_value, _, _ = stats.linregress(np.log(f_fit), np.log(s_fit))

        return (abs(slope - expected_slope) <= tolerance) and (r_value ** 2 >= 0.9)

    def check_noise(self, spectrum: np.ndarray, noise_threshold: float) -> bool:
        """Check if spectrum has acceptable noise levels"""
        noise = np.std(spectrum) / np.mean(spectrum)
        return noise <= noise_threshold


class BandpassCorrections:
    """Bandpass spectral corrections"""

    def __init__(self, sampling_freq: float):
        """
        Initialize bandpass corrections

        Args:
            sampling_freq: Sampling frequency in Hz
        """
        self.fs = sampling_freq
        self.tf = EnhancedTransferFunctions()

    def analytical_correction(self, measured_flux: float, u_star: float,
                              z: float, L: float, stability: StabilityClass,
                              transfer_params: Dict) -> float:
        """
        Calculate analytical spectral correction

        Args:
            measured_flux: Uncorrected flux
            u_star: Friction velocity
            z: Measurement height
            L: Obukhov length
            stability: Stability class
            transfer_params: Parameters for transfer functions

        Returns:
            Corrected flux
        """
        # Generate frequencies
        freqs = np.fft.fftfreq(int(self.fs * 3600))[1:int(self.fs * 3600 // 2)]

        # Get model cospectrum
        model = SpectralModels()
        cospec = model.moncrieff_cospectrum(freqs, u_star, z, L)

        # Calculate transfer functions
        tfs = []
        for name, params in transfer_params.items():
            if hasattr(self.tf, name):
                tf = getattr(self.tf, name)(freqs, **params)
                tfs.append(tf)

        # Combine transfer functions
        total_tf = self.tf.combined_transfer(freqs, tfs)

        # Calculate correction
        correction = SpectralCorrections(self.fs)
        factor = correction.calculate_correction(measured_flux, freqs,
                                                 cospec, total_tf)

        return measured_flux * factor

    def in_situ_correction(self, w_ts: np.ndarray, w_gas: np.ndarray,
                           freq_range: Tuple[float, float] = (0.01, 1.0)) -> float:
        """
        Calculate in-situ spectral correction using temperature as reference

        Args:
            w_ts: Vertical wind-temperature covariance time series
            w_gas: Vertical wind-gas covariance time series
            freq_range: Frequency range for correction

        Returns:
            Correction factor
        """
        # Calculate spectra
        spec = SpectralAnalysis(self.fs)
        f_wt, co_wt = spec.calculate_cospectra(w_ts[:, 0], w_ts[:, 1])
        f_wg, co_wg = spec.calculate_cospectra(w_gas[:, 0], w_gas[:, 1])

        # Select frequency range
        mask = (f_wt >= freq_range[0]) & (f_wt <= freq_range[1])

        # Calculate correction as ratio of cospectra
        ratio = np.trapz(co_wt[mask], f_wt[mask]) / \
                np.trapz(co_wg[mask], f_wg[mask])

        return ratio