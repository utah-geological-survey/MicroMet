LI-COR Wrapper Module
==================

.. module:: micromet.licor_wrapper

Python wrapper for running the EddyPro® software engine.

Functions
--------

.. autofunction:: run_eddypro
.. autofunction:: main

Basic Usage
---------

.. code-block:: python

    from micromet.licor_wrapper import run_eddypro

    # Basic usage with default parameters
    result = run_eddypro()

    # Run with specific project file on Windows
    result = run_eddypro(
        system="win",
        proj_file="path/to/project.eddypro"
    )

    # Run in embedded mode
    result = run_eddypro(
        mode="embedded",
        environment="/path/to/working/dir",
        proj_file="project.eddypro"
    )

Command Line Interface
-------------------

.. code-block:: bash

    python licor_wrapper.py [-h] [-s {win,linux,mac}] [-m {embedded,desktop}] 
                           [-c {gui,console}] [-e ENVIRONMENT] [proj_file]

Options:
    -h, --help            Show help message
    -s, --system          Operating system (default: win)
    -m, --mode           Running mode (default: desktop)
    -c, --caller         Caller type (default: console)
    -e, --environment    Working directory for embedded mode
    proj_file           Path to project file (optional)

Installation Requirements
----------------------

* EddyPro® software must be installed
* eddypro_rp executable must be in the system PATH
* Python 3.6+

Error Handling
------------

The module includes error handling for:

* EddyPro executable not found
* Process execution failures
* Invalid parameter combinations
* File access issues

License
-------

This wrapper is provided as-is without any warranty. EddyPro® is a registered trademark of LI-COR Biosciences.
