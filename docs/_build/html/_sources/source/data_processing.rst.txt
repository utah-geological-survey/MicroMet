Data Processing and Management
==============================

This section explains how Utah Flux Network data are processed and managed using Python.

Modules
-------

**1. `converter.py`**
- `AmerifluxDataProcessor`: Parses AmeriFlux TOA5 CSVs into pandas DataFrames.
- `Reformatter`: Cleans, standardizes, and resamples data using timestamp inference, column renaming, and soil sensor logic.

**2. `tools.py`**
- Detects irrigation events (`find_irr_dates`)
- Identifies missing data gaps and visualizes them (`find_gaps`, `plot_gaps`)
- Flags extreme variations

**3. `graphs.py`**
- `energy_sankey()`: Visualizes daily energy balance as Sankey diagrams
- `scatterplot_instrument_comparison()`: Compares instruments with regression stats

**4. `headers.py`**
- Utilities for detecting and applying missing headers across files

**5. `station_data_pull.py`**
- Fetches logger data over HTTP from remote stations
- Compares and inserts data into SQL databases
