
Classes
---------

.. toctree::
    :hidden:
    :glob:
    
    classes/floris
    classes/floris.simulation
    classes/floris.tools

.. The following classes are used in the FLORIS controls-oriented modeling tool to evaluate a wind farm.  Check out the links for more details on each of the classes contained in FLORIS.

.. Farm
.. ==========

.. Farm is the container class of the FLORIS package. It brings together all of the component objects after input (ie Turbine, Wake, FlowField) and packages everything into the appropriate data type. Farm should also be used as an entry point to probe objects for generating output.

.. Flow Field
.. =================

.. FlowField is at the core of the FLORIS package. This class handles the domain creation and initialization and computes the flow field based on the input wake model and turbine map. It also contains helper functions for quick flow field visualization.

.. Turbine Map 
.. ==============

.. TurbineMap is container object which maps a Turbine instance to a Vec3 object. This class also provides some helper methods for sorting and manipulating the turbine layout.

.. Turbine
.. ================

.. Turbine is model object representing a particular wind turbine. It is largely a container of data and parameters, but also contains method to probe properties for output.

.. Wake Velocity 
.. ======================

.. Wake velocity is a container class for the various wake velocity models.  This includes the Jensen model, the multi-zone (original FLORIS) model, the Gaussian  model, and the curl model. 

.. Wake Deflection 
.. ======================

.. Wake deflection is a container class for the various wake deflection models.  This includes the Jimenez model and the Gaussian deflection model.

.. Wake 
.. ======================

.. Wake is a container class for the various wake model objects. In particular, Wake holds references to the velocity and deflection models as well as their parameters.

.. Wake Combination
.. ======================

.. These functions return u_field with u_wake incorporated

..     u_field: the modified flow field without u_wake
    
..     u_wake: the wake to add into the rest of the flow field


