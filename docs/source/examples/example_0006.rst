example_0006_floris_demonstrate_hh_plot.py
==================================

This example demonstrates two methods for getting flow data from a FLORIS model.  The code uses the python time module to indicate the difference
in required time involved.

::

        fi.get_flow_data()


Is the conventional method and returns a 3D flow data field from which both horizontal and vertical cut throughs could be extacted, while

::

    fi.get_hub_height_flow_data()

Only returns data near hub-height, which could be useful in quickly plotting the horizontal cut through
