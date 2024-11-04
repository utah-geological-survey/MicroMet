===============
converter.py
===============

A Python module for processing and reformatting eddy covariance flux data.

.. contents:: Table of Contents
   :depth: 3

Overview
--------

The ``converter.py`` module provides tools for processing raw eddy covariance (EC) data into standardized formats
suitable for flux calculations and analysis. It handles data cleaning, quality control, and format standardization
for meteorological and flux measurements.

Main Components
--------------

Reformatter Class
~~~~~~~~~~~~~~~~

.. py:class:: Reformatter(et_data, drop_soil=True, data_path=None)

   The main class for processing and reformatting eddy covariance data.

   :param et_data: Input DataFrame containing raw EC data
   :type et_data: pandas.DataFrame
   :param drop_soil: Whether to remove extra soil parameters
   :type drop_soil: bool
   :param data_path: Path to variable limits configuration file
   :type data_path: str or pathlib.Path

   .. py:method:: datefixer(et_data)

      Fixes datetime formatting and handles duplicates in the data.

      :param et_data: Input DataFrame with timestamp columns
      :type et_data: pandas.DataFrame
      :returns: DataFrame with fixed datetime index
      :rtype: pandas.DataFrame

   .. py:method:: run_irga(df)

      Processes IRGA (Infrared Gas Analyzer) data through the complete processing pipeline.

      :param df: Input DataFrame with IRGA measurements
      :type df: pandas.DataFrame
      :returns: Series containing processed flux values
      :rtype: pandas.Series

   .. py:method:: despike(arr, nstd=4.5)

      Removes spikes from measurement data.

      :param arr: Array of measurements
      :type arr: numpy.ndarray
      :param nstd: Number of standard deviations for spike detection
      :type nstd: float
      :returns: Despiked array
      :rtype: numpy.ndarray

Utility Functions
~~~~~~~~~~~~~~~~

.. py:function:: dataframe_from_file(file)

   Reads data from a CSV file and returns a standardized DataFrame.

   :param file: Path to input file
   :type file: str or pathlib.Path
   :returns: DataFrame with standardized column names
   :rtype: pandas.DataFrame or None

.. py:function:: raw_file_compile(raw_fold, station_folder_name, search_str="*Flux_AmeriFluxFormat*.dat")

   Compiles multiple raw data files into a single DataFrame.

   :param raw_fold: Path to root folder containing raw files
   :type raw_fold: pathlib.Path
   :param station_folder_name: Name of station subfolder
   :type station_folder_name: str
   :param search_str: Pattern for matching data files
   :type search_str: str
   :returns: Combined DataFrame from all matching files
   :rtype: pandas.DataFrame

Configuration Constants
----------------------

Header Dictionaries
~~~~~~~~~~~~~~~~~~

The module includes several predefined header dictionaries for different site configurations::

    default = ['TIMESTAMP_START', 'TIMESTAMP_END', 'CO2', ...]
    bflat = list(filter(lambda item: item not in ('TA_1_1_4', ...), default))
    wellington = list(filter(lambda item: item not in 'TS_1_1_1', default))

Data Processing Pipeline
-----------------------

1. Data Loading
~~~~~~~~~~~~~~

Raw data is loaded using :func:`dataframe_from_file` or :func:`raw_file_compile`::

    df = dataframe_from_file('raw_data.csv')
    # or
    df = raw_file_compile(Path('data_dir'), 'station1')

2. Initial Processing
~~~~~~~~~~~~~~~~~~~~

Data is processed through the Reformatter class::

    reformatter = Reformatter(df)
    reformatter.name_changer()  # Standardize variable names
    reformatter.extreme_limiter(df)  # Remove invalid values

3. Quality Control
~~~~~~~~~~~~~~~~~

Several quality control steps are applied::

    - Despiking of measurements
    - Removal of physically impossible values
    - Coordinate rotation for wind components
    - SSITC scaling for quality flags

4. Final Processing
~~~~~~~~~~~~~~~~~~

Final calculations and corrections::

    results = reformatter.run_irga(df)  # Complete processing pipeline

Examples
--------

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from converter import Reformatter, dataframe_from_file

    # Load data
    df = dataframe_from_file('ec_data.csv')

    # Initialize reformatter
    reformatter = Reformatter(df)

    # Process data
    results = reformatter.run_irga(df)

    print(results['H'])  # Access sensible heat flux
    print(results['ET'])  # Access evapotranspiration

Processing Multiple Files
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from converter import raw_file_compile, Reformatter

    # Compile multiple files
    df = raw_file_compile(Path('data_dir'), 'station1')

    # Process compiled data
    reformatter = Reformatter(df)
    results = reformatter.run_irga(df)

Error Handling
-------------

The module includes several error handling mechanisms::

    - Returns None for invalid files instead of raising exceptions
    - Replaces invalid values with NaN
    - Includes boundary checking for physical parameters
    - Validates timestamp consistency

Dependencies
-----------

- pandas
- numpy
- scipy
- pathlib
- configparser

Development Notes
---------------

Testing
~~~~~~~

Run the test suite using pytest::

    pytest test_converter.py -v

Contributing
~~~~~~~~~~~

1. Follow PEP 8 style guidelines
2. Add tests for new functionality
3. Update documentation for changes
4. Use type hints for new functions

Version History
--------------

- 1.0.0 (2024-01-01)
    - Initial release
    - Basic data processing functionality
    - Support for IRGA measurements

- 1.1.0 (2024-02-01)
    - Added support for soil parameters
    - Improved coordinate rotation
    - Enhanced error handling

Future Improvements
------------------

1. Add support for additional sensor types
2. Implement parallel processing for large datasets
3. Add automated quality control reporting
4. Enhance visualization capabilities

References
----------

1. Webb, E.K., Pearman, G.I. and Leuning, R., 1980. Correction of flux measurements for density effects due to heat and water vapour transfer. Q.J.R. Meteorol. Soc., 106: 85-100.

2. Foken, T., 2008. Micrometeorology. Springer-Verlag, Berlin Heidelberg.

3. AmeriFlux Network Data Standards

Contact
-------

For issues and contributions:
https://github.com/yourusername/converter

License
-------

MIT License. See LICENSE file for details.