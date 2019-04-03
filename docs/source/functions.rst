
.. toctree::
    :hidden:
    :glob:

    source/functions/farm
    source/functions/flow_field
    source/functions/turbine
    source/functions/wake_deflection
    source/functions/wake_velocity
    source/functions/wake
    source/functions/wake_combination

Classes
---------

The following classes are used in the FLORIS controls-oriented modeling tool to evaluate a wind farm.

Farm...
==========

Farm is the container class of the FLORIS package. It brings together all of the component objects after input (ie Turbine, Wake, FlowField) and packages everything into the appropriate data type. Farm should also be used as an entry point to probe objects for generating output.

Flow Field
=================

FlowField is at the core of the FLORIS package. This class handles the domain creation and initialization and computes the flow field based on the input wake model and turbine map. It also contains helper functions for quick flow field visualization.

Turbine Map 
==============

TurbineMap is container object which maps a Turbine instance to a Vec3 object. This class also provides some helper methods for sorting and manipulating the turbine layout.

Turbine
================

Turbine is model object representing a particular wind turbine. It is largely a container of data and parameters, but also contains method to probe properties for output.

Wake Velocity 
======================

XXX

Wake Deflection 
======================

XXX

Wake 
======================

Wake is a container class for the various wake model objects. In particular, Wake holds references to the velocity and deflection models as well as their parameters.

Wake Combination
======================

XXX


