Outlier Removal Module
===================

.. module:: micromet.outlier_removal

Module for detecting and cleaning extreme variations in time series data.

Functions
--------

.. autofunction:: detect_extreme_variations
.. autofunction:: clean_extreme_variations

Example Usage
-----------

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from micromet.outlier_removal import clean_extreme_variations

    # Clean extreme variations from data
    cleaning_results = clean_extreme_variations(
        df=data,
        fields=['temperature', 'humidity'],
        frequency='D',
        variation_threshold=3.0,
        replacement_method='interpolate'
    )

    # Access results
    cleaned_data = cleaning_results['cleaned_data']
    cleaning_summary = cleaning_results['cleaning_summary']
    removed_points = cleaning_results['removed_points']

Advanced Usage
~~~~~~~~~~~~

.. code-block:: python

    # Detect extreme variations without cleaning
    variation_results = detect_extreme_variations(
        df=data,
        fields=['temperature', 'humidity'],
        frequency='D',
        variation_threshold=3.0
    )

    # Access variation information
    variations = variation_results['variations']
    extreme_points = variation_results['extreme_points']
    summary = variation_results['summary']

Configuration Options
------------------

Replacement Methods
~~~~~~~~~~~~~~~~

The following replacement methods are available:

* 'nan': Replace with NaN
* 'interpolate': Linear interpolation
* 'mean': Replace with frequency mean
* 'median': Replace with frequency median

Frequency Options
~~~~~~~~~~~~~~

Common frequency strings:

* 'D': Daily
* 'H': Hourly
* '30min': 30 minutes
* '15min': 15 minutes

Best Practices
------------

1. Data Quality
   * Check for gaps before processing
   * Use appropriate thresholds
   * Document any data modifications

2. Processing Steps
   * Choose appropriate frequency
   * Validate results
   * Keep track of removed points

3. Performance
   * Use vectorized operations
   * Consider data size
   * Monitor memory usage
