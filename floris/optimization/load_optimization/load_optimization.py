"""Module for the load optimization class and functions."""

import numpy as np

from floris import FlorisModel
from floris.core import State


def get_max_powers(fmodel: FlorisModel):
    """Get the rated power of each turbine in the farm.

    Args:
        fmodel (FlorisModel): FlorisModel object

    Returns:
        np.array: Array of rated powers for each turbine
    """
    rated_powers = np.zeros(fmodel.n_turbines)
    for t in range(fmodel.n_turbines):
        rated_powers[t] = fmodel.core.farm.turbine_map[0].power_thrust_table["power"].max() * 1000.0

    return rated_powers


def get_rotor_diameters(fmodel: FlorisModel):
    """Get the rotor diameter of each turbine

    Args:
        fmodel (FlorisModel): FlorisModel object

    Returns:
        np.array: Array of rated powers for each turbine
    """

    # If fmodel.core.farm.rotor_diameters is 1 dimensional, return
    if fmodel.core.farm.rotor_diameters.ndim == 1:
        return fmodel.core.farm.rotor_diameters
    else:
        return fmodel.core.farm.rotor_diameters[0, :]


def compute_load_ti(
    fmodel: FlorisModel,
    load_ambient_tis: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
):
    """Compute the turbine 'load' turbulence intensity for the current layout.

    Args:
        fmodel (FlorisModel): FlorisModel object
        load_ambient_tis (list or np.array): Ambient 'load' turbulence intensity for each findex
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.
    """

    if fmodel.core.state is not State.USED:
        raise ValueError("FlorisModel must be run before computing load turbulence intensity")

    rotor_diameters = get_rotor_diameters(fmodel)

    # Get the indices for sorting and unsorting
    sorted_indices = fmodel.core.grid.sorted_indices[:, :, 0, 0]
    unsorted_indices = fmodel.core.grid.unsorted_indices[:, :, 0, 0]

    # Ensure load_ambient_tis is a list or np.array
    if not isinstance(load_ambient_tis, (list, np.ndarray)):
        raise ValueError("load_ambient_tis must be a list or np.array")

    # Ensure load_ambient_tis is  of length n_findex
    if len(load_ambient_tis) != fmodel.n_findex:
        raise ValueError(
            (
                "load_ambient_tis must be a list or np.array of length n_findex",
                f"FMODEL findex = {fmodel.n_findex}, load_ambient_tis = {len(load_ambient_tis)}",
            )
        )

    # Initialize the load_ti to the load_ambient_ti
    # This should be n_findex x n_turbines
    # Tile the ambient ti across the turbines
    load_ti = np.tile(np.array(load_ambient_tis).reshape(-1, 1), (1, fmodel.n_turbines))

    # Get the turbine thrust coefficients
    # n_findex x n_turbines
    cts = fmodel.get_turbine_thrust_coefficients()

    # Get the ambient wind speeds
    # n_findex
    ambient_wind_speeds = fmodel.wind_speeds.reshape(-1, 1)

    # Reshape the ambient ti for multiplication
    load_ambient_tis_reshape = np.array(load_ambient_tis).reshape(-1, 1)

    # Get the x-sorted locations
    x_sorted = np.mean(fmodel.core.grid.x_sorted, axis=(2, 3))
    y_sorted = np.mean(fmodel.core.grid.y_sorted, axis=(2, 3))

    # Put ct into sorted frame
    ct_sorted = np.take_along_axis(cts, sorted_indices, axis=1)

    # 2. Iterate over turbines from front to back across findices
    for t in range(fmodel.n_turbines):
        # Set D to the diameter of the current turbine
        D = rotor_diameters[t]

        # Get current turbine locations
        x_t = x_sorted[:, t].reshape(-1, 1)
        y_t = y_sorted[:, t].reshape(-1, 1)

        # Get the current ct value
        ct_t = ct_sorted[:, t].reshape(-1, 1)

        # Get the differences
        dx = x_sorted - x_t
        dy = y_sorted - y_t  # Note no deflection

        # Set the boundary mask
        wake_cone_mask = np.abs(dy) < D + wake_slope * dx

        # Set the downstream mask
        downstream_mask = dx > 0

        # Calculate total distance
        distance = np.sqrt(dx**2 + dy**2)

        # Set the minimum distance mask
        max_dist_mask = distance < D * max_dist_D

        # Compute the standard deviation of the wind speed owed to this wake
        ws_std = np.where(
            wake_cone_mask & downstream_mask & max_dist_mask,
            ambient_wind_speeds / (1.5 + 0.8 * (distance / D) / np.sqrt(ct_t)),
            0.0,
        )

        # Combine with ambient TI to get the total TI from this wake
        ti_add_update = (
            np.sqrt(ws_std**2 + (load_ambient_tis_reshape * ambient_wind_speeds) ** 2)
            / ambient_wind_speeds
        )

        # Update the load_ti using maximum wake TI
        load_ti = np.maximum(load_ti, ti_add_update)

    # Re-sort load_ti to non-sorted frame
    load_ti = np.take_along_axis(load_ti, unsorted_indices, axis=1)

    return load_ti


def compute_turbine_voc(
    fmodel: FlorisModel,
    A: float,
    load_ambient_tis: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Compute the turbine Variable Operating Cost (VOC) for each findex and turbine.

    In this first approximation, variable operating cost is computed as the
    product of turbine thrust and wind speed standard deviation.

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        load_ambient_tis (list or np.array): Ambient 'load' turbulence intensity for each findex
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.0.

    Returns:
        np.array: Array of VOC for each findex and turbine
    """

    # A should be a float and not a list or array
    if not isinstance(A, (int, float)):
        raise ValueError("A (coefficient of VOC) must be a float")

    # Get the ambient wind speed and apply to each turbine per findex
    ambient_wind_speeds = fmodel.wind_speeds
    ambient_wind_speeds = np.tile(ambient_wind_speeds[:, np.newaxis], (1, fmodel.n_turbines))

    # Compute the rotor area
    D = fmodel.core.farm.rotor_diameters[0, 0]
    area = np.pi * (D / 2) ** 2

    # Compute the thrust
    cts = fmodel.get_turbine_thrust_coefficients()
    thrust = 0.5 * fmodel.core.flow_field.air_density * area * cts * ambient_wind_speeds**2

    # Compute the load_ti
    load_ti = compute_load_ti(
        fmodel=fmodel,
        load_ambient_tis=load_ambient_tis,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
    )

    # Compute wind speed standard deviation
    ws_std = ambient_wind_speeds * load_ti

    # Compute voc
    return A * (ws_std**exp_ws_std) * (thrust**exp_thrust)


def compute_farm_revenue(
    fmodel: FlorisModel,
):
    """Compute the farm revenue of the FlorisModel object.

    Args:
        fmodel (FlorisModel): FlorisModel object

    Returns:
        np.array: Array of farm revenue for each findex

    """

    if fmodel.core.state is not State.USED:
        raise ValueError("FlorisModel must be run before computing net revenue")

    # Make sure fmodel.wind_data is not None
    if fmodel.wind_data is None:
        raise ValueError("FlorisModel must have wind data to compute net revenue")

    # Ensure that fmodel.wind_data is not None
    if fmodel.wind_data.values is None:
        raise ValueError("FlorisModel wind_data.values must be set to compute revenue")

    farm_power = fmodel.get_farm_power()
    values = fmodel.wind_data.values
    return farm_power * values


def compute_farm_voc(
    fmodel: FlorisModel,
    A: float,
    load_ambient_tis: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Compute the farm Variable Operating Cost (VOC) for each findex.

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        load_ambient_tis (list or np.array): Ambient 'load' turbulence intensity for each findex
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.0.

    Returns:
        np.array: Array of farm VOC for each findex

    """
    turbine_voc = compute_turbine_voc(
        fmodel=fmodel,
        A=A,
        load_ambient_tis=load_ambient_tis,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )
    return np.sum(turbine_voc, axis=1)


def compute_net_revenue(
    fmodel: FlorisModel,
    A: float,
    load_ambient_tis: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Compute the net revenue for the current layout."

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        load_ambient_tis (list or np.array): Ambient 'load' turbulence intensity for each findex
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.0.

    Returns:
        np.array: Array of net revenue for each findex

    """

    revenue = compute_farm_revenue(
        fmodel=fmodel,
    )

    farm_voc = compute_farm_voc(
        fmodel=fmodel,
        A=A,
        load_ambient_tis=load_ambient_tis,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    return revenue - farm_voc


def find_A_to_satisfy_rev_voc_ratio(
    fmodel: FlorisModel,
    target_rev_voc_ratio: float,
    load_ambient_tis: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Find the value of A that satisfies the target revenue to VOC ratio.

    Args:
        fmodel (FlorisModel): FlorisModel object
        target_rev_voc_ratio (float): Target revenue to VOC ratio
        load_ambient_tis (list or np.array): Ambient 'load' turbulence intensity for each findex
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.

    Returns:
        float: Value of A that satisfies the target revenue to VOC ratio

    """

    # Compute farm revenue
    farm_revenue = compute_farm_revenue(
        fmodel=fmodel,
    )

    # Compute farm VOC
    farm_voc = compute_farm_voc(
        fmodel=fmodel,
        A=1.0,
        load_ambient_tis=load_ambient_tis,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    return (farm_revenue.sum() / farm_voc.sum()) / target_rev_voc_ratio


def optimize_derate(
    fmodel: FlorisModel,
    A: float,
    load_ambient_tis: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
    initial_power_setpoint: np.array = None,
    derating_levels: np.array = np.linspace(1.0, 0.001, 5),
):
    """Optimize the derating of each turbine to maximize net revenue.

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        load_ambient_tis (list or np.array): Ambient 'load' turbulence intensity for each findex
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.
        initial_power_setpoint (np.array, optional): Initial power setpoint for each turbine.
            Defaults to None.
        derating_levels (np.array, optional): Array of derating levels to consider.
            Defaults to np.linspace(1.0, 0.001, 5).

    """

    # Ensure we're in derating mode
    fmodel.set_operation_model("simple-derating")

    # If initial set point not provided, set to rated (assumed max) power
    if initial_power_setpoint is None:
        max_power = get_max_powers(fmodel)
        initial_power_setpoint = np.tile(max_power, (fmodel.n_findex, 1))

    # Initialize the the test power setpoints
    test_power_setpoint = initial_power_setpoint.copy()
    final_power_setpoint = initial_power_setpoint.copy()

    # Get the sorted coords
    sorted_indices = fmodel.core.grid.sorted_indices[:, :, 0, 0]

    # Initialize the net revenue using the initial setpoints
    fmodel.set(power_setpoints=initial_power_setpoint)
    fmodel.run()
    net_revenue = compute_net_revenue(
        fmodel=fmodel,
        A=A,
        load_ambient_tis=load_ambient_tis,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    # Now loop over turbines
    for t in sorted_indices.T:
        # Loop over derating levels
        for d in derating_levels:
            # Apply the proposed derating level to the test_power_setpoint matrix
            test_power_setpoint[range(fmodel.n_findex), t] = (
                initial_power_setpoint[range(fmodel.n_findex), t] * d
            )

            # Apply the setpoint to fmodel
            fmodel.set(power_setpoints=test_power_setpoint)

            # Run
            fmodel.run()

            # Get the net revenue
            test_net_revenue = compute_net_revenue(
                fmodel=fmodel,
                A=A,
                load_ambient_tis=load_ambient_tis,
                wake_slope=wake_slope,
                max_dist_D=max_dist_D,
            )

            # Get a map of where test_net_revenue is greater than net_revenue
            update_mask = test_net_revenue > net_revenue

            # print(f"Derating {d} turbine {t} for {np.sum(update_mask)} findex")
            # print(f"Net: {net_revenue}")
            # print(f"Test: {test_net_revenue}")
            # print(f"Mask: {update_mask}")

            # Where update_mask is false, revert the test_power_setpoint to previous value
            test_power_setpoint[~update_mask, :] = final_power_setpoint[~update_mask, :]

            # Update the final_power_setpoint
            final_power_setpoint[:, :] = test_power_setpoint[:, :]

            # Update the net_revenue
            net_revenue[update_mask] = test_net_revenue[update_mask]

    # Return the final power setpoint
    return final_power_setpoint
