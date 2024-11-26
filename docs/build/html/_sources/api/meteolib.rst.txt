Meteorological Library Module
==========================

.. module:: micromet.meteolib

A modern Python library for meteorological calculations.

Classes
-------

MeteoCalculator
~~~~~~~~~~~~~

.. autoclass:: MeteoCalculator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

MeteoConfig
~~~~~~~~~~

.. autoclass:: MeteoConfig
   :members:
   :undoc-members:
   :show-inheritance:

Constants
--------

.. code-block:: python

    GRAVITY = 9.81            # Acceleration due to gravity [m/s^2]
    VON_KARMAN = 0.41        # von Karman constant
    STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]

Example Usage
-----------

Basic Calculations
~~~~~~~~~~~~~~~~

.. code-block:: python

    from micromet.meteolib import MeteoCalculator

    calc = MeteoCalculator()

    # Calculate specific heat
    cp = calc.specific_heat(temp=25.0, rh=60.0, pressure=101300.0)
    print(f"Specific heat: {cp:.2f} J kg⁻¹ K⁻¹")

    # Calculate vapor pressure slope
    delta = calc.vapor_pressure_slope(temp=25.0)
    print(f"Vapor pressure slope: {delta:.2f} Pa K⁻¹")

Advanced Features
~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate Penman-Monteith reference ET
    et = calc.penman_monteith_reference(
        airtemp=25.0,
        rh=60.0,
        airpress=101300.0,
        rs=70.0,
        rn=500.0,
        g=50.0,
        u2=2.0
    )

Unit Handling
-----------

The module supports multiple temperature units:

.. code-block:: python

    from micromet.meteolib import TemperatureUnit, MeteoConfig

    # Configure calculator with specific temperature unit
    config = MeteoConfig(temp_unit=TemperatureUnit.CELSIUS)
    calc = MeteoCalculator(config)

Error Handling
------------

The module includes comprehensive error checking:

* Input validation for physical limits
* Handling of non-finite values
* Unit conversion validation
* Appropriate warning messages for out-of-range values

References
---------

* FAO-56 Penman-Monteith equations
* Buck (1981) equations for vapor pressure
* World Meteorological Organization standards
