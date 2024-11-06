"""
Plotting functionality for eddy covariance spectral analysis.
Provides specialized plots for spectra, cospectra, transfer functions,
and quality control visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from .spectral import SpectralParameters
from .spectral import StabilityClass

class SpectralPlots:
    """Class for creating spectral and cospectral plots"""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize plotting class
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    def plot_spectrum(self, freqs: np.ndarray, spectrum: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     normalize: bool = True,
                     var_name: str = '',
                     show_slope: bool = True,
                     **kwargs) -> plt.Axes:
        """
        Plot power spectrum
        
        Args:
            freqs: Frequency array
            spectrum: Spectral values
            ax: Matplotlib axes to plot on
            normalize: Normalize by frequency if True
            var_name: Variable name for label
            show_slope: Show -5/3 reference line
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Normalize if requested
        if normalize:
            spectrum = spectrum * freqs
            
        # Plot spectrum
        ax.loglog(freqs, spectrum, **kwargs)
        
        # Add -5/3 reference line if requested
        if show_slope:
            f_range = np.array([freqs[0], freqs[-1]])
            ref_line = f_range**(-5/3) * np.max(spectrum) * 10
            ax.loglog(f_range, ref_line, '--', color='gray', 
                     label='-5/3 slope', alpha=0.7)
            
        # Labels
        ax.set_xlabel('Frequency [Hz]')
        if normalize:
            ax.set_ylabel(f'f·S{var_name}(f)')
        else:
            ax.set_ylabel(f'S{var_name}(f)')
            
        # Grid
        ax.grid(True, which='both', alpha=0.3)
        
        return ax
    
    def plot_cospectrum(self, freqs: np.ndarray, cospec: np.ndarray,
                       ax: Optional[plt.Axes] = None,
                       normalize: bool = True,
                       var_names: Tuple[str, str] = ('',''),
                       show_slope: bool = True,
                       **kwargs) -> plt.Axes:
        """
        Plot cospectrum
        
        Args:
            freqs: Frequency array
            cospec: Cospectral values
            ax: Matplotlib axes to plot on
            normalize: Normalize by frequency if True
            var_names: Tuple of variable names for label
            show_slope: Show -7/3 reference line
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Normalize if requested    
        if normalize:
            cospec = cospec * freqs
            
        # Plot cospectrum
        ax.semilogx(freqs, cospec, **kwargs)
        
        # Add -7/3 reference line if requested
        if show_slope:
            f_range = np.array([freqs[0], freqs[-1]])
            ref_line = f_range**(-7/3) * np.max(cospec) * 10
            ax.loglog(f_range, ref_line, '--', color='gray', 
                     label='-7/3 slope', alpha=0.7)
            
        # Labels
        ax.set_xlabel('Frequency [Hz]')
        v1, v2 = var_names
        if normalize:
            ax.set_ylabel(f'f·Co{v1}{v2}(f)')
        else:
            ax.set_ylabel(f'Co{v1}{v2}(f)')
            
        # Grid
        ax.grid(True, which='both', alpha=0.3)
        
        return ax
    
    def plot_ogive(self, freqs: np.ndarray, ogive: np.ndarray,
                  ax: Optional[plt.Axes] = None,
                  var_names: Tuple[str, str] = ('',''),
                  **kwargs) -> plt.Axes:
        """
        Plot ogive (cumulative cospectrum)
        
        Args:
            freqs: Frequency array
            ogive: Ogive values
            ax: Matplotlib axes to plot on
            var_names: Tuple of variable names for label
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Plot ogive
        ax.semilogx(freqs, ogive, **kwargs)
        
        # Labels
        ax.set_xlabel('Frequency [Hz]')
        v1, v2 = var_names
        ax.set_ylabel(f'Og{v1}{v2}(f)')
        
        # Grid
        ax.grid(True, which='both', alpha=0.3)
        
        return ax
    
    def plot_transfer_functions(self, freqs: np.ndarray, 
                              transfer_funcs: Dict[str, np.ndarray],
                              ax: Optional[plt.Axes] = None,
                              **kwargs) -> plt.Axes:
        """
        Plot transfer functions
        
        Args:
            freqs: Frequency array
            transfer_funcs: Dictionary of transfer functions
            ax: Matplotlib axes to plot on
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Plot each transfer function
        for name, tf in transfer_funcs.items():
            ax.semilogx(freqs, tf, label=name, **kwargs)
            
        # Labels
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Transfer Function')
        ax.legend()
        
        # Grid
        ax.grid(True, which='both', alpha=0.3)
        
        return ax
    
    def diagnostic_plots(self, freqs: np.ndarray, spectrum: np.ndarray,
                        fitted_params: SpectralParameters,
                        qc_results: Dict[str, bool]) -> None:
        """
        Create diagnostic plots for spectral analysis
        
        Args:
            freqs: Frequency array
            spectrum: Spectral values
            fitted_params: Fitted spectral parameters
            qc_results: Dictionary of QC test results
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)
        
        # Original and fitted spectra
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_spectrum(freqs, spectrum, ax=ax1, label='Measured')
        
        # Generate fitted spectrum
        fitted_spec = (freqs/fitted_params.f_peak) / \
                     (1 + (freqs/fitted_params.f_peak)**(fitted_params.broadness * fitted_params.slope))
        ax1.loglog(freqs, fitted_spec, '--', label='Fitted')
        ax1.legend()
        
        # Residuals
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = spectrum - fitted_spec
        ax2.semilogx(freqs, residuals)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
        
        # QC summary
        ax3 = fig.add_subplot(gs[1, :])
        qc_items = list(qc_results.items())
        x_pos = np.arange(len(qc_items))
        colors = ['green' if v else 'red' for _, v in qc_items]
        
        ax3.bar(x_pos, [1]*len(qc_items), color=colors)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([k for k, _ in qc_items], rotation=45)
        ax3.set_ylabel('Pass/Fail')
        ax3.set_title('Quality Control Results')
        
        plt.tight_layout()
        
    def stability_comparison(self, freqs: np.ndarray, 
                           spectra: Dict[StabilityClass, np.ndarray]) -> None:
        """
        Plot spectra across stability classes
        
        Args:
            freqs: Frequency array
            spectra: Dictionary of spectra by stability class
        """
        _, ax = plt.subplots(figsize=(8, 6))
        
        colors = {
            StabilityClass.VERY_UNSTABLE: 'red',
            StabilityClass.UNSTABLE: 'orange',
            StabilityClass.NEUTRAL: 'gray',
            StabilityClass.STABLE: 'lightblue',
            StabilityClass.VERY_STABLE: 'blue'
        }
        
        for stability, spectrum in spectra.items():
            ax.loglog(freqs, spectrum * freqs, 
                     color=colors[stability],
                     label=stability.name.replace('_', ' '))
            
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('f·S(f)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
    def correction_summary(self, freqs: np.ndarray,
                         orig_cospec: np.ndarray,
                         corrected_cospec: np.ndarray,
                         transfer_func: np.ndarray) -> None:
        """
        Summary plot of spectral corrections
        
        Args:
            freqs: Frequency array
            orig_cospec: Original cospectrum
            corrected_cospec: Corrected cospectrum
            transfer_func: Transfer function used
        """
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Plot cospectra
        self.plot_cospectrum(freqs, orig_cospec, ax=ax1, 
                           label='Original', color='blue')
        self.plot_cospectrum(freqs, corrected_cospec, ax=ax1,
                           label='Corrected', color='red')
        ax1.legend()
        
        # Plot transfer function
        ax2.semilogx(freqs, transfer_func, color='black')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Transfer Function')
        ax2.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
