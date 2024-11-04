MeteoLib Documentation
=====================

Overview
--------

MeteoLib is a Python library for meteorological calculations, with a focus on evapotranspiration and related atmospheric physics. It provides implementations of standard meteorological equations and methods following established scientific literature.

Installation
-----------

.. code-block:: bash

    pip install meteolib

Quick Start
----------

.. code-block:: python

    from meteolib import MeteoCalculator

    # Create calculator instance
    calc = MeteoCalculator()

    # Calculate reference evapotranspiration
    ET0 = calc.ET0pm(
        airtemp=25.0,      # Air temperature [°C]
        rh=60.0,           # Relative humidity [%]
        airpress=101300.0, # Air pressure [Pa]
        Rs=20e6,           # Solar radiation [J m-2 day-1]
        Rext=40e6,         # Extraterrestrial radiation [J m-2 day-1]
        u=2.5             # Wind speed [m s-1]
    )

Core Features
------------

* Multiple evapotranspiration estimation methods:

  * Penman (1948, 1956) open water evaporation
  * Makkink (1965) evaporation
  * Priestley-Taylor (1972) evaporation
  * FAO Penman-Monteith reference evapotranspiration

* Psychrometric calculations
* Solar radiation and atmospheric physics
* Input validation and error handling
* Support for both scalar and array inputs

Main Components
-------------

MeteoCalculator Class
~~~~~~~~~~~~~~~~~~~~

The primary interface for all meteorological calculations. Handles input validation and provides a consistent API for all methods.

.. code-block:: python

    calc = MeteoCalculator(config=MeteoConfig())

Configuration
~~~~~~~~~~~~

.. code-block:: python

    from meteolib import MeteoConfig, TemperatureUnit

    config = MeteoConfig(
        temp_unit=TemperatureUnit.CELSIUS,
        validate_inputs=True,
        raise_warnings=True
    )

Evapotranspiration Methods
-------------------------

Penman Open Water Evaporation (E0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: E0(airtemp, rh, airpress, Rs, Rext, u, alpha=0.08, Z=0.0)

    Calculate Penman (1948, 1956) open water evaporation.

    :param airtemp: Daily average air temperature [°C]
    :type airtemp: float or array_like
    :param rh: Daily average relative humidity [%]
    :type rh: float or array_like
    :param airpress: Daily average air pressure [Pa]
    :type airpress: float or array_like
    :param Rs: Daily incoming solar radiation [J m-2 day-1]
    :type Rs: float or array_like
    :param Rext: Daily extraterrestrial radiation [J m-2 day-1]
    :type Rext: float or array_like
    :param u: Daily average wind speed at 2m [m s-1]
    :type u: float or array_like
    :param alpha: Albedo [-], defaults to 0.08 for open water
    :type alpha: float, optional
    :param Z: Site elevation [m]
    :type Z: float, optional
    :returns: Open water evaporation [mm day-1]
    :rtype: float or ndarray

Makkink Evaporation (Em)
~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: Em(airtemp, rh, airpress, Rs)

    Calculate Makkink (1965) evaporation.

    :param airtemp: Daily average air temperature [°C]
    :type airtemp: float or array_like
    :param rh: Daily average relative humidity [%]
    :type rh: float or array_like
    :param airpress: Daily average air pressure [Pa]
    :type airpress: float or array_like
    :param Rs: Average daily incoming solar radiation [J m-2 day-1]
    :type Rs: float or array_like
    :returns: Makkink evaporation [mm day-1]
    :rtype: float or ndarray

Priestley-Taylor Evaporation (Ept)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: Ept(airtemp, rh, airpress, Rn, G)

    Calculate Priestley-Taylor (1972) evaporation.

    :param airtemp: Daily average air temperature [°C]
    :type airtemp: float or array_like
    :param rh: Daily average relative humidity [%]
    :type rh: float or array_like
    :param airpress: Daily average air pressure [Pa]
    :type airpress: float or array_like
    :param Rn: Average daily net radiation [J m-2 day-1]
    :type Rn: float or array_like
    :param G: Average daily soil heat flux [J m-2 day-1]
    :type G: float or array_like
    :returns: Priestley-Taylor evaporation [mm day-1]
    :rtype: float or ndarray

FAO Penman-Monteith Reference ET (ET0pm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: ET0pm(airtemp, rh, airpress, Rs, Rext, u, Z=0.0)

    Calculate FAO Penman-Monteith reference evaporation for short grass.

    :param airtemp: Daily average air temperature [°C]
    :type airtemp: float or array_like
    :param rh: Daily average relative humidity [%]
    :type rh: float or array_like
    :param airpress: Daily average air pressure [Pa]
    :type airpress: float or array_like
    :param Rs: Daily incoming solar radiation [J m-2 day-1]
    :type Rs: float or array_like
    :param Rext: Extraterrestrial radiation [J m-2 day-1]
    :type Rext: float or array_like
    :param u: Wind speed at 2m [m s-1]
    :type u: float or array_like
    :param Z: Elevation [m]
    :type Z: float, optional
    :returns: Reference evapotranspiration [mm day-1]
    :rtype: float or ndarray

Auxiliary Functions
-----------------

Psychrometric Calculations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate psychrometric constant
    gamma = calc.psychrometric_constant(airtemp, rh, airpress)

    # Calculate vapor pressure slope
    delta = calc.vapor_pressure_slope(airtemp)

    # Calculate actual vapor pressure
    ea = calc.actual_vapor_pressure(airtemp, rh)

    # Calculate saturation vapor pressure
    es = calc.saturation_vapor_pressure(airtemp)

Solar Calculations
~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate solar parameters
    results = calc.solar_parameters(doy=180, lat=45.0)
    max_sunshine_hours = results.max_sunshine_hours
    extraterrestrial_radiation = results.extraterrestrial_radiation

Physical Limits and Validation
---------------------------

The library enforces physical limits on input parameters:

+------------+-----------+---------+-------+
| Parameter  | Minimum   | Maximum | Units |
+============+===========+=========+=======+
| airtemp    | -273.15   | 100     | °C    |
+------------+-----------+---------+-------+
| rh         | 0         | 100     | %     |
+------------+-----------+---------+-------+
| airpress   | 1000      | 120000  | Pa    |
+------------+-----------+---------+-------+

Error Handling
-------------

The library uses custom exceptions for error handling:

.. code-block:: python

    class MeteoError(Exception):
        """Base exception class for meteorological calculation errors"""
        pass

Typical error scenarios:

* Invalid input values (outside physical limits)
* Non-finite values (NaN or Inf)
* Mismatched array dimensions
* Invalid parameter combinations

Working with Arrays
-----------------

The library supports NumPy arrays for batch processing:

.. code-block:: python

    import numpy as np

    # Create array inputs
    temps = np.array([20.0, 25.0, 30.0])
    rh = np.array([50.0, 60.0, 70.0])

    # Calculate for multiple conditions at once
    results = calc.ET0pm(temps, rh, ...)

Best Practices
------------

1. Input Validation:
   
   * Enable input validation during development
   * Consider disabling for production if performance is critical

2. Unit Consistency:
   
   * Always use SI units as specified in the documentation
   * Pay special attention to radiation units (J m-2 day-1)

3. Error Handling:
   
   * Always catch MeteoError exceptions
   * Check input arrays for consistency

4. Performance:
   
   * Use array operations for batch processing
   * Consider using the same calculator instance for multiple calculations

Examples
-------

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from meteolib import MeteoCalculator

    calc = MeteoCalculator()

    # Calculate reference ET
    ET0 = calc.ET0pm(
        airtemp=25.0,      # Air temperature [°C]
        rh=60.0,           # Relative humidity [%]
        airpress=101300.0, # Air pressure [Pa]
        Rs=20e6,           # Solar radiation [J m-2 day-1]
        Rext=40e6,         # Extraterrestrial radiation [J m-2 day-1]
        u=2.5,            # Wind speed [m s-1]
        Z=100.0           # Elevation [m]
    )

Batch Processing
~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    # Create arrays of conditions
    temps = np.linspace(20, 30, 24)  # Temperature range
    rhs = np.ones_like(temps) * 60   # Constant RH
    u = np.ones_like(temps) * 2.5    # Constant wind speed

    # Calculate ET0 for all conditions
    ET0_array = calc.ET0pm(
        airtemp=temps,
        rh=rhs,
        airpress=101300.0,
        Rs=20e6,
        Rext=40e6,
        u=u
    )

Error Handling
~~~~~~~~~~~~

.. code-block:: python

    try:
        result = calc.ET0pm(
            airtemp=-300,  # Invalid temperature
            rh=60.0,
            airpress=101300.0,
            Rs=20e6,
            Rext=40e6,
            u=2.5
        )
    except MeteoError as e:
        print(f"Calculation error: {e}")

References
---------

1. Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass. Proc. Roy. Soc. London, A193, 120-146.
2. Makkink, G.F. (1965). A comparison of some methods to estimate evapotranspiration from grass fields.
3. Priestley, C.H.B. and R.J. Taylor (1972). On the assessment of surface heat flux and evaporation.
4. Allen, R.G., et al. (1998). FAO Irrigation and Drainage Paper No. 56 - Crop Evapotranspiration.

License
-------

MIT License - See LICENSE file for details.

