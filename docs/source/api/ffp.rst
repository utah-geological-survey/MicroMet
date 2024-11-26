Flux Footprint Module
===================

.. module:: micromet.ffp

A module for calculating and analyzing flux footprints with support for coordinate transformations and georeferencing.

Classes
-------

FootprintInput
~~~~~~~~~~~~~

.. autoclass:: FootprintInput
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

CoordinateSystem
~~~~~~~~~~~~~~

.. autoclass:: CoordinateSystem
   :members:
   :undoc-members:
   :show-inheritance:

FootprintCalculator
~~~~~~~~~~~~~~~~~

.. autoclass:: FootprintCalculator
   :members:
   :undoc-members:
   :show-inheritance:

EnhancedFootprintProcessor
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EnhancedFootprintProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
------------

Basic Footprint Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from micromet.ffp import FootprintInput, FootprintCalculator

    # Create input parameters
    inputs = FootprintInput(
        zm=3.0,          # Measurement height above displacement height (z-d) [m]
        z0=0.1,          # Roughness length [m]
        umean=2.5,       # Mean wind speed at zm [ms-1]
        h=1000.0,        # Boundary layer height [m]
        ol=-50.0,        # Obukhov length [m]
        sigmav=0.5,      # Standard deviation of lateral velocity fluctuations [ms-1]
        ustar=0.3,       # Friction velocity [ms-1]
        wind_dir=180.0   # Wind direction in degrees
    )

    # Initialize calculator and compute footprint
    calculator = FootprintCalculator()
    result = calculator.calculate_footprint(inputs)

Georeferenced Footprint
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from micromet.ffp import (FootprintConfig, CoordinateSystem, 
                            EnhancedFootprintProcessor)

    # Setup coordinate systems
    source_crs = CoordinateSystem.from_epsg(4326)  # WGS84
    working_crs = CoordinateSystem.from_epsg(32612)  # UTM Zone 12N

    # Create configuration
    config = FootprintConfig(
        origin_distance=1000.0,
        measurement_height=3.0,
        roughness_length=0.1,
        domain_size=(-1000, 1000, -1000, 1000),
        grid_resolution=20.0,
        station_coords=(-111.0, 41.0),
        coordinate_system=source_crs,
        working_crs=working_crs
    )

    # Process footprint
    processor = EnhancedFootprintProcessor(config)
    result = processor.calculate_georeferenced_footprint(footprint_input)

References
---------

Kljun, N., Calanca, P., Rotach, M.W., Schmid, H.P., 2015. A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP). Geosci. Model Dev. 8, 3695-3713.
