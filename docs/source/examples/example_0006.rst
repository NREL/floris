example_0006_floris_demonstrate_hh_plot.py
==========================================

The code for this example can be found here: `example_0006_floris_demonstrate_hh_plot.py
<https://github.com/NREL/floris/blob/develop/examples/example_0006_floris_demonstrate_hh_plot.py>`_

This example demonstrates two methods for getting flow data from a FLORIS 
model. The code uses the python time module to indicate the difference in 
required time involved.

::

        fi.get_flow_data()


:py:meth:`get_flow_data() 
<floris.tools.floris_interface.FlorisInterface.get_flow_data>` is the 
conventional method and returns a 3D flow data field from which both horizontal 
and vertical cut throughs could be extacted.

::

    fi.get_hub_height_flow_data()

:py:meth:`get_hub_height_flow_data() 
<floris.tools.floris_interface.FlorisInterface.get_hub_height_flow_data>` only 
returns data near hub-height, which could be useful in quickly plotting the 
horizontal cut through.
