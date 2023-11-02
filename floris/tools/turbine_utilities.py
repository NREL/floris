import pandas as pd
import numpy as np
# import yaml or something? How do I print, format a yaml?

def build_turbine_yaml(
    turbine_data,
    turbine_name,
    file_path=None,
    generator_efficiency=1.0,
    hub_height=90.0,
    pP=1.88,
    pT=1.88,
    rotor_diameter=126.0,
    TSR=8.0,
    ref_density_cp_ct=1.225,
    ref_tilt_cp_ct=5.0
):
    """
    Tool for formatting a turbine yaml from data formatted as a pandas dataframe.

    Default value for turbine physical parameters are from the NREL 5MW reference
    wind turbine.

    turbine_data is a pandas DataFrame that contains columns specifying the 
    turbine power and thrust as a function of wind speed. The following columns
    are possible:
    - wind_speed [m/s]
    - power_absolute [kW]
    - power_coefficient [-]
    - thrust_absolute [kN] 
    - thrust_coefficient [-]
    Of these, wind_speed is required. One of power_absolute and power_coefficient
    must be specified; and one of thrust_absolute and thrust_coefficient must be
    specified. If both _absolute and _coeffient versions are specified, the 
    _coefficient entry will be used and the _absolute entry ignored.

    Args:
        turbine_data (pd.DataFrame): Dataframe containing performance of the wind
           turbine as a function of wind speed. Described in more detail above.
        turbine_name (string): Name of the turbine, which will be used for the
           turbine_type field as well as the filename.
        file_path (): Path for placement of the produced yaml. Defaults to None,
           which places the turbine in the running directory.
        generator_efficiency (float): Generator efficiency [-]. Defaults to 1.0.
        hub_height (float): Hub height [m]. Defaults to 90.0.
        pP (float): Cosine exponent for power loss to yaw [-]. Defaults to 1.88.
        pT (float): Cosine exponent for thrust loss to yaw [-]. Defaults to 1.88.
        rotor_diameter (float). Rotor diameter [m]. Defaults to 126.0.
        TSR (float). Turbine optimal tip-speed ratio [-]. Defaults to 8.0.
        ref_density_cp_ct (float). Air density used to specify power and thrust 
            curves [kg/m^3]. Defaults to 1.225.
        ref_tilt_cp_ct (float). Rotor tilt (due to shaft tilt and/or platform 
            tilt) used when defining the power and thrust curves [deg]. Defaults
            to 5.0.
    """

    # Check that necessary columns are specified
    if "wind_speed" not in turbine_data.columns:
        raise KeyError("wind_speed column must be specified.")
    
    # Construct the Cp curve
    if "power_coefficient" in turbine_data.columns:
        if "power_absolute" in turbine_data.columns:
            print(
                "Found both power_absolute and power_coefficient."
                "Ignoring power_absolute."
            )
        Cp = turbine_data.power_coefficient.values
    elif "power_absolute" in turbine_data.columns:
        # Check units look ok.
        pass # Conversion to Cp goes here.
    else:
       raise KeyError(
           "Either power_absolute or power_coefficient must be specified."
        )


    # Construct Ct curve
    if "thrust_coefficient" in turbine_data.columns:
        if "thrust_absolute" in turbine_data.columns:
            print(
                "Found both thrust_absolute and thrust_coefficient."
                "Ignoring thrust_absolute."
            )
        Ct = turbine_data.thrust_coefficient.values
    elif "thrust_absolute" in turbine_data.columns:
        # Check units look ok.
        pass # Conversion to Cp goes here.
    else:
       raise KeyError(
           "Either thrust_absolute or thrust_coefficient must be specified."
        )



    return None