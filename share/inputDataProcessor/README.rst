FLORIS Input File Generation from the United States Wind Turbine Database
-------------------------------------------------------------------------

The ``farm_processor.py`` script generates FLORIS input files from the United States Wind Turbine Database.  It runs on the CSV data that can be downloaded from https://eerscmap.usgs.gov/uswtdb/data/.

More information on the metadata can be found here: https://eerscmap.usgs.gov/uswtdb/assets/data/uswtdb_v1_0_20180419.xml

Usage
=========================
For usage help: 

.. code-block:: python

  python farm_processor.py -h


To process CSV data and generate input files:

.. code-block:: python

  python farm_processor.py -a <path_to_csv_data_file>

Dependencies
============
The following packages are required for FLORIS

- Python3

- utm (pip install utm)
