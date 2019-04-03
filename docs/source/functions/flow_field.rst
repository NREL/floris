
Flow Field
------------

FlowField is at the core of the FLORIS package. This class handles the domain
creation and initialization and computes the flow field based on the input
wake model and turbine map. It also contains helper functions for quick flow
field visualization.

Inputs 
=========

wind_speed: float - atmospheric condition

wind_direction - atmospheric condition

wind_shear - atmospheric condition

wind_veer - atmospheric condition

turbulence_intensity - atmospheric condition

wake: Wake - used to calculate the flow field

wake_combination: WakeCombination - used to combine turbine wakes into the flow field

turbine_map: TurbineMap - locates turbines in space 

Outputs
=========

self: FlowField - an instantiated FlowField object

Members
=========

``floris.farm.flow_field.reinitialize_flow_field(self,
                                wind_speed=None,
                                wind_direction=None,
                                wind_shear=None,
                                wind_veer=None,
                                turbulence_intensity=None,
                                air_density=None,
                                wake=None,
                                turbine_map=None,
                                with_resolution=None)``

``calculate_wake``

Getters/Setters 
===================

wind_direction

domain_bounds




