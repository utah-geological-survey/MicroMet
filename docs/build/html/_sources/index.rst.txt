Welcome to Micromet's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/ec
   api/converter
   api/ffp
   api/meteolib
   api/outlier_removal
   api/tools
   api/graphs
   api/cs_wrapper
   api/licor_wrapper
   examples/basic_usage
   examples/advanced_usage

Micromet is a comprehensive Python package for processing and analyzing micrometeorological data,
with a focus on eddy covariance flux measurements.

Key Features
-----------

- Eddy covariance flux calculations with standard corrections
- AmeriFlux format data handling and conversion
- Flux footprint modeling with coordinate system support
- Data quality control and outlier removal
- Meteorological calculations and unit conversions
- Visualization tools for flux and energy balance data
- Integration with Campbell Scientific and LI-COR instruments

Quick Start
----------

Installation
^^^^^^^^^^^

.. code-block:: bash

   pip install micromet

Basic Usage
^^^^^^^^^^

.. code-block:: python

   import micromet
   from micromet.ec import CalcFlux
   from micromet.converter import Reformatter

   # Load and format raw data
   raw_data = micromet.converter.raw_file_compile('raw_data_path', 'station_name')
   reformatter = Reformatter(raw_data)

   # Calculate fluxes
   flux_calculator = CalcFlux()
   results = flux_calculator.runall(reformatter.et_data)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
