Campbell Scientific Wrapper Module
==============================

.. module:: micromet.cs_wrapper

Python wrapper for Campbell Scientific's csidft_convert.exe utility.

Functions
--------

.. autofunction:: convert_file
.. autofunction:: main

Basic Usage
---------

.. code-block:: python

    from micromet.cs_wrapper import convert_file

    # Basic conversion to TOA5 format
    convert_file('input.dat', 'output.csv', 'toa5')

    # Array-based conversion with FSL file
    convert_file(
        'input.dat', 
        'output.csv', 
        'toa5',
        fsl_file='input.fsl',
        array_id='1'
    )

Command Line Usage
---------------

.. code-block:: bash

    python cs_wrapper.py input.dat output.csv toa5 --fsl input.fsl --array 1

Output Formats
------------

Supported output formats:

* 'toaci1': Array-compatible CSV format version 1
* 'toa5': Table-oriented ASCII format version 5
* 'tob1': Binary table format version 1
* 'csixml': Campbell Scientific XML format
* 'custom-csv': Custom CSV format
* 'no-header': CSV with no header information

Installation Requirements
----------------------

1. Campbell Scientific LoggerNet software must be installed
2. csidft_convert.exe must be in the system PATH or specified via exe_path

Error Handling
------------

The module includes error handling for:

* File not found errors
* Conversion process failures
* Invalid parameter combinations
* System path issues

References
---------

For more information on Campbell Scientific file formats and conversion options, see:
https://help.campbellsci.com/loggernet-manual/ln_manual/campbell_scientific_file_formats/csidft_convert.exe.htm
