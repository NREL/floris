example_0020_modify_floris_to_match.py
======================================

The code for this example can be found here: 
`example_0020_modify_floris_to_match.py 
<https://github.com/NREL/floris/blob/develop/examples/example_0020_modify_floris_to_match.py>`_

This example follows the previous example in loading a saved SOWFA case.  In 
this example, a FLORIS case is adjusted to match SOWFA to demonstrate this 
process.

The first section of codes repeats the process of loading interfaces to both 
SOWFA and FLORIS files and producing hub-height cut-throughs.  Then data from 
the SOWFA interface is used to set the FLORIS model to match in this code.

::

    # Set the relevant FLORIS parameters to equal the SOWFA case
    fi.reinitialize_flow_field(wind_speed=si.precursor_wind_speed,
                    wind_direction=si.precursor_wind_dir,
                    layout_array=(si.layout_x, si.layout_y)
                    )
    # Set the yaw angles
    fi.calculate_wake(yaw_angles=si.yaw_angles)


The resultant FLORIS flow field is trimmed to match SOWFA here:

:: 

    # Generate and get a flow from original FLORIS file
    floris_flow_data_matched = fi.get_flow_data()

    # Trim the flow to match SOWFA
    sowfa_domain_limits = [
        [np.min(sowfa_flow_data.x), np.max(sowfa_flow_data.x)],
        [np.min(sowfa_flow_data.y), np.max(sowfa_flow_data.y)],
        [np.min(sowfa_flow_data.z), np.max(sowfa_flow_data.z)]]
    floris_flow_data_matched = floris_flow_data_matched.crop(
        floris_flow_data_matched, sowfa_domain_limits[0],
        sowfa_domain_limits[1], sowfa_domain_limits[2])