Quickstart Guide
==============

This guide will help you get started with the Micromet package for processing and analyzing micrometeorological data.

Installation
-----------

You can install Micromet using pip:

.. code-block:: bash

    pip install micromet

For development installation:

.. code-block:: bash

    git clone https://github.com/inkenbrandt/micromet.git
    cd micromet
    pip install -e .

Basic Usage
----------

Processing Raw Data
~~~~~~~~~~~~~~~~

Start by importing the necessary modules and loading your data:

.. code-block:: python

    import micromet
    from micromet.converter import Reformatter, raw_file_compile
    
    # Load raw data files from a directory
    raw_data = raw_file_compile(
        raw_fold='path/to/data',
        station_folder_name='STATION1'
    )
    
    # Convert to AmeriFlux format
    reformatter = Reformatter(raw_data)
    formatted_data = reformatter.et_data

Quality Control
~~~~~~~~~~~~~

Apply quality control measures to your data:

.. code-block:: python

    from micromet.tools import find_gaps, clean_extreme_variations
    
    # Find data gaps
    gaps = find_gaps(
        formatted_data,
        columns=['temperature', 'humidity', 'pressure']
    )
    
    # Clean extreme variations
    cleaned_data = clean_extreme_variations(
        formatted_data,
        frequency='D',
        variation_threshold=3.0,
        replacement_method='interpolate'
    )['cleaned_data']

Flux Calculations
~~~~~~~~~~~~~~~

Calculate fluxes using the eddy covariance method:

.. code-block:: python

    from micromet.ec import CalcFlux
    
    # Initialize flux calculator
    calculator = CalcFlux(
        UHeight=3.0,
        sonic_dir=240
    )
    
    # Calculate fluxes
    results = calculator.runall(cleaned_data)
    
    # Access calculated values
    sensible_heat = results['H']
    latent_heat = results['lambdaE']
    friction_velocity = results['Ustr']

Footprint Analysis
~~~~~~~~~~~~~~~~

Calculate flux footprints:

.. code-block:: python

    from micromet.ffp import FootprintInput, FootprintCalculator
    
    # Create input parameters
    inputs = FootprintInput(
        zm=3.0,          # Measurement height
        z0=0.1,          # Roughness length
        umean=2.5,       # Mean wind speed
        h=1000.0,        # Boundary layer height
        ol=-50.0,        # Obukhov length
        sigmav=0.5,      # Standard deviation of lateral velocity
        ustar=0.3,       # Friction velocity
        wind_dir=180.0   # Wind direction
    )
    
    # Calculate footprint
    calculator = FootprintCalculator()
    footprint = calculator.calculate_footprint(inputs)

Visualization
-----------

Create energy balance diagrams:

.. code-block:: python

    from micromet.graphs import energy_sankey
    
    # Create Sankey diagram
    fig = energy_sankey(results, date_text="2024-06-19 12:00")
    fig.show()

Plot time series data:

.. code-block:: python

    from micromet.graphs import plot_timeseries_daterange
    
    # Create time series plot
    plot_timeseries_daterange(
        cleaned_data,
        selected_station='STATION1',
        selected_field='temperature',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

Hardware Integration
-----------------

Campbell Scientific Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert Campbell Scientific data files:

.. code-block:: python

    from micromet.cs_wrapper import convert_file
    
    # Convert to TOA5 format
    convert_file(
        'input.dat',
        'output.csv',
        'toa5'
    )

LI-COR Integration
~~~~~~~~~~~~~~~

Run EddyPro processing:

.. code-block:: python

    from micromet.licor_wrapper import run_eddypro
    
    # Run EddyPro with specific project file
    result = run_eddypro(
        system="win",
        proj_file="path/to/project.eddypro"
    )

Common Workflows
-------------

Complete Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete workflow combining multiple features:

.. code-block:: python

    import pandas as pd
    from micromet.converter import Reformatter, raw_file_compile
    from micromet.tools import clean_extreme_variations
    from micromet.ec import CalcFlux
    from micromet.graphs import energy_sankey
    
    # 1. Load and format data
    raw_data = raw_file_compile('data_path', 'STATION1')
    reformatter = Reformatter(raw_data)
    
    # 2. Clean data
    cleaned_data = clean_extreme_variations(
        reformatter.et_data,
        frequency='D',
        variation_threshold=3.0,
        replacement_method='interpolate'
    )['cleaned_data']
    
    # 3. Calculate fluxes
    calculator = CalcFlux()
    results = calculator.runall(cleaned_data)
    
    # 4. Visualize results
    fig = energy_sankey(results)
    fig.show()

Common Issues
-----------

Missing Data
~~~~~~~~~~
Handle missing data appropriately:

.. code-block:: python

    # Find gaps in data
    gaps = find_gaps(data, columns=['temperature'])
    
    # Interpolate missing values
    data = data.interpolate(method='time', limit=24)

Timezone Handling
~~~~~~~~~~~~~~
Ensure consistent timezone handling:

.. code-block:: python

    # Convert to UTC
    data.index = data.index.tz_localize('UTC')
    
    # Convert to local time
    data.index = data.index.tz_convert('America/Denver')

Next Steps
---------

- Check out the :doc:`API Reference </api/index>` for detailed documentation
- Review :doc:`Examples </examples/index>` for more use cases
- Visit the `GitHub repository <https://github.com/utah-geological-survey/MicroMet>`_ for source code
- Report issues on the `Issue Tracker <https://github.com/utah-geological-survey/MicroMet/issues>`_

Support
------

For questions and issues:

- GitHub Issues: https://github.com/utah-geological-survey/MicroMet/issues
- Documentation: https://micromet.readthedocs.io
- Email: paulinkenbrandt@utah.gov
