EC Module
========

.. module:: micromet.ec

The EC module provides core eddy covariance calculations with standard corrections.

Classes
-------

CalcFlux
~~~~~~~~

.. autoclass:: CalcFlux
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Key Features
-----------

* Shadow correction for CSAT sonic anemometer measurements
* Coordinate rotation for wind components
* Maximum covariance calculations
* WPL (Webb-Pearman-Leuning) corrections
* Spectral corrections following Massman (2000, 2001)
* Various meteorological parameter calculations

Example Usage
------------

.. code-block:: python

    from micromet.ec import CalcFlux

    # Initialize calculator
    calc = CalcFlux(UHeight=2.0, sonic_dir=240)

    # Load high-frequency data
    data = pd.read_csv('flux_data.csv')

    # Process fluxes
    results = calc.runall(data)

    # Access processed values
    sensible_heat = results['H']
    latent_heat = results['lambdaE']
    friction_velocity = results['Ustr']

Physical Constants
----------------

The module includes several important physical constants:

.. code-block:: python

    Rv = 461.51    # Water vapor gas constant (J/kg·K)
    Ru = 8.3143    # Universal gas constant (J/kg·K)
    Cpd = 1005.0   # Specific heat of dry air (J/kg·K)
    Rd = 287.05    # Dry air gas constant (J/kg·K)
    g = 9.81       # Gravitational acceleration (m/s²)
    k = 0.41       # von Karman constant

References
---------

* Webb, E.K., Pearman, G.I., and Leuning, R. (1980): Correction of flux measurements for density effects due to heat and water vapour transfer. Q.J.R. Meteorol. Soc., 106, 85-100.
* Massman, W.J. (2000): A simple method for estimating frequency response corrections for eddy covariance systems. Agricultural and Forest Meteorology, 104, 185-198.
* Schotanus, P., Nieuwstadt, F.T.M., and de Bruin, H.A.R. (1983): Temperature measurement with a sonic anemometer and its application to heat and moisture fluxes. Boundary-Layer Meteorology, 26, 81-93.
