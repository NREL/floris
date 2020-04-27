Inputs
------

.. contents:: Contents
    :local:
    :backlinks: none

Configuration files for FLORIS are stored in the ``.json`` format, which is
then parsed by :py:class:`floris.simulation.input_reader.InputReader` and
stored in python dictionaries. The file begins with file type, name of the
FLORIS configuration, and a brief description of the FLORIS configuration:

::

    "type": "floris_input",
    "name": "floris_input_file_Example",
    "description": "Example FLORIS Input file",
    "floris_version": "v2.0.0,

Conversion Utility
==================
FLORIS contains an input file conversion tool to update older files with the
latest API changes. This utility is included as a Python command-line program
in `floris/share/preprocessor`. It is executed by "running" the directory
itself, and the syntax can be obtained by passing the "help" flag:

.. code-block:: python

    # From the floris/share directory
    python preprocessor -h

An example usage is to convert an input file from `v1.1.0` to the latest,
`v2.0.0`:

.. code-block:: python

    python preprocessor -i big_farm.json -o big_farm_v2.json

This utility can convert older files to a newer version of FLORIS, but not
the other way. It can also be used to generate a starter input file containing
default values for the latest supported version by running with no arguments.

Model Inputs
============
The input file contains model information in three primary sections:

- :py:class:`~.turbine.Turbine`
- :py:class:`~.wake.Wake`
- :py:class:`~.farm.Farm`

The ``Wake`` section describes the wake models and their corresponding
parameters. ``Turbine`` configures the turbine geometry and performance
parameters. Finally, the ``Farm`` section describes the layout of the turbines
within the farm and the atmospheric conditions throughout.

For model-specifics required in each section, see the example input files
and corresponding class documentation.

.. _sample_input_file_ref:

Sample Input File
=================

A series of example input files are available in ``examples/other_jsons``.
The most basic comprehensive example is given at ``examples/example_input.json``
and included below. It describes a 2x2 wind turbine array modeling the
NREL 5-MW wind turbine.

.. literalinclude:: ../../examples/example_input.json
