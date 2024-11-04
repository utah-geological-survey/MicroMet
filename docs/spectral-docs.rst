====================================
EC Spectral Analysis Documentation
====================================

A Python library for spectral analysis of eddy covariance data, including calculation of power spectra, co-spectra, 
and comparison with theoretical models.

Features
--------
- Power spectral density calculation
- Co-spectral analysis
- Kaimal spectrum comparison
- Block averaging
- Inertial subrange analysis
- Normalized frequency plotting
- Synthetic data generation

Installation
------------

Dependencies
~~~~~~~~~~~
- numpy
- scipy
- pandas
- matplotlib

Basic Usage
----------

.. code-block:: python

    import pandas as pd
    from ec_spectral import spectral_analysis, plot_wt_cospectra
    
    # Generate example data or use your own DataFrame
    df = generate_example_ec_data()
    
    # Perform spectral analysis
    results = spectral_analysis(df)
    
    # Create w'T' cospectra plot with theoretical spectrum
    fig = plot_wt_cospectra(df)

API Reference
------------

spectral_analysis
~~~~~~~~~~~~~~~~
.. py:function:: spectral_analysis(df, variables=['Ux', 'Uy', 'Uz', 'Ts', 'pV'], sampling_freq=20)

    Perform spectral analysis on eddy covariance data.

    :param df: DataFrame containing high frequency eddy covariance data
    :type df: pandas.DataFrame
    :param variables: List of variable names to analyze
    :type variables: list
    :param sampling_freq: Sampling frequency in Hz
    :type sampling_freq: float
    :return: Dictionary containing spectral analysis results
    :rtype: dict
    
    The returned dictionary contains:
    
    - 'spectra': Power spectra for each variable
    - 'cospectra': Co-spectra between vertical wind and other variables
    - 'frequencies': Frequency arrays
    - 'normalized_freqs': Normalized frequencies (f*z/U)
    - 'peaks': Spectral peak information
    - 'dissipation_rate': Energy dissipation rates

plot_wt_cospectra
~~~~~~~~~~~~~~~~
.. py:function:: plot_wt_cospectra(df, sampling_freq=20, z=3.0, u_star=0.5, L=-50, show_slope=True, slope=-2/3)

    Plot w'T' cospectra with theoretical Kaimal spectrum and optional slope line.

    :param df: DataFrame with 'Uz' and 'Ts' columns
    :type df: pandas.DataFrame
    :param sampling_freq: Sampling frequency in Hz
    :type sampling_freq: float
    :param z: Measurement height (m)
    :type z: float
    :param u_star: Friction velocity (m/s)
    :type u_star: float
    :param L: Obukhov length (m)
    :type L: float
    :param show_slope: Whether to show slope reference line
    :type show_slope: bool
    :param slope: Slope value to show
    :type slope: float
    :return: Figure object
    :rtype: matplotlib.figure.Figure

calc_kaimal_spectrum
~~~~~~~~~~~~~~~~~~~
.. py:function:: calc_kaimal_spectrum(f, z=3.0, u_star=0.5, L=-50)

    Calculate theoretical Kaimal spectrum for w'T' cospectra.

    :param f: Frequencies in Hz
    :type f: array_like
    :param z: Measurement height (m)
    :type z: float
    :param u_star: Friction velocity (m/s)
    :type u_star: float
    :param L: Obukhov length (m)
    :type L: float
    :return: Normalized cospectral density values
    :rtype: array_like

generate_example_ec_data
~~~~~~~~~~~~~~~~~~~~~~~
.. py:function:: generate_example_ec_data(duration=30, sampling_freq=20, include_noise=True, seed=None)

    Generate synthetic eddy covariance data for testing and demonstration.

    :param duration: Duration of time series in minutes
    :type duration: float
    :param sampling_freq: Sampling frequency in Hz
    :type sampling_freq: float
    :param include_noise: Whether to add random noise
    :type include_noise: bool
    :param seed: Random seed for reproducibility
    :type seed: int, optional
    :return: DataFrame with synthetic EC data
    :rtype: pandas.DataFrame

Examples
--------

Basic Spectral Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Generate example data
    df = generate_example_ec_data(duration=30, sampling_freq=20)
    
    # Perform spectral analysis
    results = spectral_analysis(df)
    
    # Print mean spectral properties
    for var in results['spectra']:
        print(f"{var} peak frequency: {results['peaks'][var]['freq'][0]:.3f} Hz")

Kaimal Spectrum Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create plot with theoretical spectrum
    fig = plot_wt_cospectra(df, z=3.0, u_star=0.5, L=-50)
    
    # Add -2/3 slope reference line
    fig = plot_wt_cospectra(df, show_slope=True, slope=-2/3)

Theory
------

Spectral Analysis
~~~~~~~~~~~~~~~~
The spectral analysis is based on Fourier decomposition of turbulent signals into their frequency components. 
The power spectrum represents the distribution of variance across frequencies, while the co-spectrum represents 
the frequency distribution of covariance between two signals.

Kaimal Spectrum
~~~~~~~~~~~~~~
The Kaimal spectrum provides a theoretical model for the behavior of turbulent cospectra under different stability 
conditions. For unstable conditions, the normalized cospectrum follows:

.. math::

    \frac{f Co_{wT}}{u_* T_*} = \frac{14n}{(1 + 9.6n)^{2.4}}

where :math:`n = fz/U` is the normalized frequency.

Inertial Subrange
~~~~~~~~~~~~~~~~
In the inertial subrange, spectral energy follows a power law decay with slope -2/3 for temperature cospectra. 
This behavior is universal and can be used to verify the quality of measurements.

References
----------
1. Kaimal, J.C., and J.J. Finnigan, 1994: Atmospheric Boundary Layer Flows: Their Structure and Measurement. 
   Oxford University Press, 289 pp.

2. Stull, R.B., 1988: An Introduction to Boundary Layer Meteorology. Kluwer Academic Publishers, 666 pp.

Contributing
-----------
Contributions are welcome! Please feel free to submit a Pull Request.

License
-------
This project is licensed under the MIT License.
