
Turbine Map
--------------

TurbineMap is container object which maps a Turbine instance to a Vec3 object. This class also provides some helper methods for sorting and manipulating the turbine layout.

Inputs 
==========

Turbine_map_dict: dict - a dictionary mapping of Turbines to Vec3.  See types for information on Vec3.

Outputs
=========

TurbineMap - an instantiated TurbineMap object 

Methods
=========

:: 

    rotated(angle, center_of_rotation)

This function is used to rotate the wind farm to due west, i.e. 270 degrees, to allow for easier computations within the wind farm.  

angle: wind direction with respect to due west, i.e. 270 degrees. 

center of rotation: center of the wind farm.

::

    sorted_in_x_as_list()

This function is used to sort the turbines from west to east.  This helps with processing turbines from upstream, with respect to winds from the west, to downstream.

Getters/Setters 
================

turbines

coords

items
