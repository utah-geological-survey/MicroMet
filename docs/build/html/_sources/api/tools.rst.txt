Tools Module
===========

.. module:: micromet.tools

The tools module provides general data processing and quality control utilities for micrometeorological data.

Functions
--------

find_irr_dates
~~~~~~~~~~~~~

.. autofunction:: find_irr_dates

Example usage:

.. code-block:: python

    from micromet.tools import find_irr_dates

    # Find irrigation dates
    dates_of_irr, swc_during_irr = find_irr_dates(
        df,
        swc_col="SWC_1_1_1",
        do_plot=True,
        dist=20,
        height=30,
        prom=0.6
    )

find_gaps
~~~~~~~~

.. autofunction:: find_gaps

Example usage:

.. code-block:: python

    from micromet.tools import find_gaps

    # Find gaps in multiple columns
    gaps_df = find_gaps(
        df,
        columns=['temperature', 'humidity', 'pressure'],
        missing_value=-9999,
        min_gap_periods=1
    )

plot_gaps
~~~~~~~~

.. autofunction:: plot_gaps

Example usage:

.. code-block:: python

    from micromet.tools import plot_gaps

    # Create an interactive visualization of data gaps
    fig = plot_gaps(gaps_df, title="Time Series Data Gaps")
    fig.show()

detect_extreme_variations
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: detect_extreme_variations

Example usage:

.. code-block:: python

    from micromet.tools import detect_extreme_variations

    # Detect extreme variations in data
    variation_results = detect_extreme_variations(
        df=data,
    