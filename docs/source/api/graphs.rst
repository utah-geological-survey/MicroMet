Graphs Module
===========

.. module:: micromet.graphs

Module for creating visualizations of flux and meteorological data.

Functions
--------

.. autofunction:: energy_sankey
.. autofunction:: bland_altman_plot
.. autofunction:: plot_timeseries_daterange
.. autofunction:: save_plot

Energy Balance Visualization
-------------------------

.. code-block:: python

    from micromet.graphs import energy_sankey

    # Create Sankey diagram
    fig = energy_sankey(flux_data, date_text="2024-06-19 12:00")
    fig.show()

Required columns for energy_sankey:

* SW_IN: Incoming shortwave radiation
* LW_IN: Incoming longwave radiation
* SW_OUT: Outgoing shortwave radiation
* LW_OUT: Outgoing longwave radiation
* NETRAD: Net radiation
* G: Ground heat flux
* LE: Latent heat
* H: Sensible heat

Method Comparison
--------------

.. code-block:: python

    from micromet.graphs import bland_altman_plot

    # Create Bland-Altman plot
    md, sd, mean, ci_low, ci_high = bland_altman_plot(
        data1, 
        data2, 
        marker='o',
        alpha=0.5
    )

Time Series Analysis
-----------------

.. code-block:: python

    from micromet.graphs import plot_timeseries_daterange

    # Create time series plot
    plot_timeseries_daterange(
        input_df=data,
        selected_station='STATION1',
        selected_field='temperature',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

Saving Plots
----------

.. code-block:: python

    from micromet.graphs import save_plot

    # Save current plot
    save_plot(None)  # Saves as 'plot.png'

Dependencies
----------

* plotly
* matplotlib
* pandas
* numpy

Customization Options
------------------

The module supports various customization options through matplotlib and plotly:

* Colors and color schemes
* Markers and line styles
* Axis labels and titles
* Figure size and resolution
* Legend position and style
