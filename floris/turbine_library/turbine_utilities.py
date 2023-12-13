import os.path

import numpy as np
import yaml


def build_turbine_dict(
    turbine_data_dict,
    turbine_name,
    file_path=None,
    generator_efficiency=1.0,
    hub_height=90.0,
    pP=1.88,
    pT=1.88,
    rotor_diameter=126.0,
    TSR=8.0,
    air_density=1.225,
    ref_tilt_cp_ct=5.0
):
    """
    Tool for formatting a full turbine dict from data formatted as a
    dictionary.

    Default value for turbine physical parameters are from the NREL 5MW reference
    wind turbine.

    Returns a turbine dictionary object as expected by FLORIS. Optionally,
    prints the dictionary to a yaml to be included in a FLORIS wake model yaml.

    turbine_data is a dictionary that contains keys specifying the
    turbine power and thrust as a function of wind speed. The following keys
    are possible:
    - wind_speed [m/s]
    - power_absolute [kW]
    - power_coefficient [-]
    - thrust_absolute [kN]
    - thrust_coefficient [-]
    Of these, wind_speed is required. One of power_absolute and power_coefficient
    must be specified; and one of thrust_absolute and thrust_coefficient must be
    specified. If both _absolute and _coefficient versions are specified, the
    _coefficient entry will be used and the _absolute entry ignored.

    Args:
        turbine_data_dict (dict): Dictionary containing performance of the wind
            turbine as a function of wind speed. Described in more detail above.
        turbine_name (string): Name of the turbine, which will be used for the
            turbine_type field as well as the filename.
        file_path (): Path for placement of the produced yaml. Defaults to None,
            in which case no yaml is written.
        generator_efficiency (float): Generator efficiency [-]. Defaults to 1.0.
        hub_height (float): Hub height [m]. Defaults to 90.0.
        pP (float): Cosine exponent for power loss to yaw [-]. Defaults to 1.88.
        pT (float): Cosine exponent for thrust loss to yaw [-]. Defaults to 1.88.
        rotor_diameter (float). Rotor diameter [m]. Defaults to 126.0.
        TSR (float). Turbine optimal tip-speed ratio [-]. Defaults to 8.0.
        air_density (float). Air density used to specify power and thrust
            curves [kg/m^3]. Defaults to 1.225.
        ref_tilt_cp_ct (float). Rotor tilt (due to shaft tilt and/or platform
            tilt) used when defining the power and thrust curves [deg]. Defaults
            to 5.0.

    Returns:
        turbine_dict (dict): Formatted turbine dictionary as expected by FLORIS.
    """

    # Check that necessary columns are specified
    if "wind_speed" not in turbine_data_dict:
        raise KeyError("wind_speed column must be specified.")
    u = np.array(turbine_data_dict["wind_speed"])
    A = np.pi * rotor_diameter**2/4

    # Construct the Cp curve
    if "power_coefficient" in turbine_data_dict:
        if "power_absolute" in turbine_data_dict:
            print(
                "Found both power_absolute and power_coefficient."
                "Ignoring power_absolute."
            )
        Cp = np.array(turbine_data_dict["power_coefficient"])

    elif "power_absolute" in turbine_data_dict:
        P = np.array(turbine_data_dict["power_absolute"])
        if _find_nearest_value_for_wind_speed(P, u, 10) > 20000 or \
           _find_nearest_value_for_wind_speed(P, u, 10) < 1000:
           print(
               "Unusual power value detected. Please check that power_absolute",
               "is specified in kW."
           )

        validity_mask = (P != 0) | (u != 0)
        Cp = np.zeros_like(P, dtype=float)

        Cp[validity_mask] = (P[validity_mask]*1000) / \
                            (0.5*air_density*A*u[validity_mask]**3)

    else:
        raise KeyError(
            "Either power_absolute or power_coefficient must be specified."
        )

    # Construct Ct curve
    if "thrust_coefficient" in turbine_data_dict:
        if "thrust_absolute" in turbine_data_dict:
            print(
                "Found both thrust_absolute and thrust_coefficient."
                "Ignoring thrust_absolute."
            )
        Ct = np.array(turbine_data_dict["thrust_coefficient"])

    elif "thrust_absolute" in turbine_data_dict:
        T = np.array(turbine_data_dict["thrust_absolute"])
        if _find_nearest_value_for_wind_speed(T, u, 10) > 3000 or \
           _find_nearest_value_for_wind_speed(T, u, 10) < 100:
           print(
               "Unusual thrust value detected. Please check that thrust_absolute",
               "is specified in kN."
           )

        validity_mask = (T != 0) | (u != 0)
        Ct = np.zeros_like(T)

        Ct[validity_mask] = (T[validity_mask]*1000)/\
                            (0.5*air_density*A*u[validity_mask]**2)

    else:
        raise KeyError(
            "Either thrust_absolute or thrust_coefficient must be specified."
        )

    # Build the turbine dict
    power_thrust_dict = {
        "wind_speed": u.tolist(),
        "power": Cp.tolist(),
        "thrust": Ct.tolist()
    }

    turbine_dict = {
        "turbine_type": turbine_name,
        "generator_efficiency": generator_efficiency,
        "hub_height": hub_height,
        "pP": pP,
        "pT": pT,
        "rotor_diameter": rotor_diameter,
        "TSR": TSR,
        "ref_density_cp_ct": air_density,
        "ref_tilt_cp_ct": ref_tilt_cp_ct,
        "power_thrust_table": power_thrust_dict
    }

    # Create yaml file
    if file_path is not None:
        full_name = os.path.join(file_path, turbine_name+".yaml")
        yaml.dump(
            turbine_dict,
            open(full_name, "w"),
            sort_keys=False
        )

        print(full_name, "created.")

    return turbine_dict

def _find_nearest_value_for_wind_speed(test_vals, ws_vals, ws):
    errs = np.absolute(ws_vals-ws)
    idx = errs.argmin()
    return test_vals[idx]
