Converter Module
==============

.. module:: micromet.converter

The Converter module provides functionality for processing and reformatting raw Utah Flux Network data into AmeriFlux compatible format.

Classes
-------

Reformatter
~~~~~~~~~~

.. autoclass:: Reformatter
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Functions
--------

.. autofunction:: raw_file_compile
.. autofunction:: dataframe_from_file
.. autofunction:: remove_extra_soil_params

Usage Example
------------

.. code-block:: python

    from micromet.converter import Reformatter, raw_file_compile

    # Compile raw data files
    raw_data = raw_file_compile('raw_data_path', 'station_name')

    # Create reformatter instance
    reformatter = Reformatter(raw_data)

    # Access reformatted data
    reformatted_data = reformatter.et_data

Configuration
------------

The module can be configured using a config.ini file with the following structure:

.. code-block:: ini

    [DEFAULT]
    pw = password
    ip = ip_address
    login = login_credentials

Data Format Requirements
----------------------

The module expects input data to follow these conventions:

* Time-indexed data with regular intervals
* Standard column names for meteorological variables
* AmeriFlux-compatible units
* Quality flags following AmeriFlux standards

Error Handling
-------------

The module includes error handling for:

* File not found errors
* Invalid header formats
* Missing credentials
* Data format inconsistencies
