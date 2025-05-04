Getting Started with Micromet
=============================

Micromet is a Python package for processing micrometeorological data from environmental monitoring stations. It provides utilities for formatting, validating, and analyzing data from sources such as AmeriFlux and Campbell Scientific systems.

This guide walks you through installation, basic usage, and how to start working with your own data.

Installation
------------

Micromet is available on PyPI and can be installed using pip:
.. code-block:: bash

    pip install micromet

Alternatively, if you want to use the latest development version, you can install it directly from GitHub:
.. code-block:: bash

    pip install git+https://github.com/inkenbrandt/MicroMet.git

MicroMet is also available as a conda package. You can install it using the following command:
.. code-block:: bash

    conda install -c conda-forge micromet


Micromet can be installed from source. First, clone the repository:

.. code-block:: bash

    git clone https://github.com/inkenbrandt/MicroMet.git
    cd MicroMet

It's recommended to use a virtual environment:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

To install in editable/development mode:

.. code-block:: bash

    pip install -e .

Usage Overview
--------------

Micromet consists of modular tools for reformatting and analyzing micrometeorological data. Here's a basic example to get started:

.. code-block:: python

    from micromet.converter import Reformatter

    # Load your raw data into a pandas DataFrame
    import pandas as pd
    df = pd.read_csv("path/to/your/data.csv")

    # Create a Reformatter instance and process the data
    ref = Reformatter()
    tidy_df = ref.prepare(df)

    # Now tidy_df contains the cleaned and normalized data
    print(tidy_df.head())

Modules
-------

The key modules in Micromet are:

- ``converter`` — Reformat raw CSV files into tidy dataframes
- ``tools`` — Utility functions for timestamp alignment and filtering
- ``headers`` — Functions for renaming columns and interpreting header metadata
- ``station_data_pull`` — (Optional) Pull and organize station-specific data files

.. note::

   The ``Notebooks`` directory is intended for exploratory analysis and is not part of the core API documentation.

Contributing
------------

We welcome contributions! If you have suggestions, bug reports, or would like to add features:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

Please make sure to add unit tests for new functionality and follow PEP8 standards.

Further Reading
---------------

- :doc:`api`
- :doc:`usage_examples`
- `Micromet on GitHub <https://github.com/inkenbrandt/MicroMet>`_

