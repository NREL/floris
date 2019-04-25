example_0020_modify_floris_to_match.py
==================================

This example follows example the previous example in loading a saved SOWFA case.  In this example, a FLORIS case is adjusted to match SOWFA to demonstrate
this process.

The first section of codes repeats the process of loading interfaces to both SOWFA and FLORIS files and producing hub-height cut-throughs.  Then data 
from the sowfa interface is used to set the FLORIS model to match in this code

::

    # Set the relevant FLORIS parameters to equal the SOWFA case
    fi.floris.farm.flow_field.reinitialize_flow_field(wind_speed=si.precursor_wind_speed,wind_direction=si.precursor_wind_dir)
    fi.floris.farm.set_turbine_locations(si.layout_x, si.layout_y, calculate_wake=False)
    fi.floris.farm.set_yaw_angles(si.yaw_angles, calculate_wake=False)


The resultant FLORIS flow field is trimmed to match SOWFA here

:: 

    # Trim the flow to match SOWFA
    sowfa_domain_limits = [[np.min(sowfa_flow_field.x), np.max(sowfa_flow_field.x)],
                        [np.min(sowfa_flow_field.y), np.max(sowfa_flow_field.y)], 
                        [np.min(sowfa_flow_field.z), np.max(sowfa_flow_field.z)]]
    floris_flow_field_matched = floris_flow_field_matched.crop(floris_flow_field_matched, sowfa_domain_limits[0], sowfa_domain_limits[1], sowfa_domain_limits[2] )

