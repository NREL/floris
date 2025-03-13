"""Module for the load optimization class and functions."""

import numpy as np

from floris import FlorisModel


def compute_load_ti(fmodel: FlorisModel,
                    load_ambient_ti: float,
                    wake_slope: float = 0.3,
                    max_dist_D: float = 10.0):
    """Compute the turbine load and turbulence intensity for the current layout.

    Args:
        fmodel (FlorisModel): FlorisModel object
        load_ambient_ti (float): Ambient turbulence intensity for loads calculation
        wake_slope (float, optional): Wake slope. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance in rotor diameters. Defaults to 10.0.

    """

    # TODO for now assume the first one
    D = fmodel.core.farm.rotor_diameters[0][0]
    print(D)


    # Initialize the load_ti to the load_ambient_ti
    # This should be _findex x n_turbines
    load_ti = np.ones((fmodel.n_findex, fmodel.n_turbines)) * load_ambient_ti

    # Get the turbine thrust coefficients
    # n_findex x n_turbines
    cts = fmodel.get_turbine_thrust_coefficients()
    print(cts)

    # Get the wind speeds
    # n_findex
    wind_speeds = fmodel.wind_speeds
    print(wind_speeds)

    # Probable steps

    # 1. Get the x-sorted locations

    # 2. Iterate over turbines from front to back across findices
    for t_i in range(fmodel.n_turbines):
        print(t_i)

        # 3. Compute dx, dy and dz for all turbines
        # Something like:
        # dx = ne.evaluate("x - x_i")
        # dy = ne.evaluate("y - y_i - deflection_field_i")
        # dz = ne.evaluate("z - z_i")


        # 4. Set the boundary mask
        # Something combining Jensen:
        # boundary_mask = ne.evaluate("sqrt(dy ** 2 + dz ** 2) < we * dx + rotor_radius")
        # Versus iea version I had coded
        # if y_diff_abs > D + x_diff * wake_slope:

        # 5. Set the max distance mask
        # if x_diff >= max_dist_D * D:

        # 6 Compute the proposed load_ti
        # From Jensen
        # c = np.where(
        #     np.logical_and(downstream_mask, boundary_mask),
        #     ne.evaluate("(rotor_radius / (rotor_radius + we * dx + NUM_EPS)) ** 2"),
        #     0.0,
        # )
        # But more like
        # Compute the added turbulence
        # ti_add_update = ambient_ws / (1.5 + 0.8 * (x_diff/D) / np.sqrt(c_t_j) )

        # # Update the ti_term
        # ti_add_update[findex, t_i] = (

        #     np.sqrt(ti_add**2 + (load_ambient_ti * ambient_ws)**2) / ambient_ws

        # )

        # 7. Update load_ti with any values which are larger
        # load_ti = np.maximum(load_ti, ti_add_update)



    return load_ti
