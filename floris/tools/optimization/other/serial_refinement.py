# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# Serial Refine method for yaw optimization
# Adaptation of Boolean Yaw Optimization by PJ Stanley
# Update with reference


import numpy as np


def _serial_refine_single_pass(fi, yaw_grid, yaw_angles_template,
                              include_unc=False, unc_pmfs=None,
                              unc_options=None, turbine_weights=None,
                              turbs_to_opt=None):
     # Get a list of the turbines in order of x and sort front to back
    layout_x = fi.layout_x
    layout_y = fi.layout_y
    wind_direction = fi.floris.farm.wind_direction[0]
    layout_x_rot = (
        np.cos((wind_direction - 270.0) * np.pi / 180.0) * layout_x
        - np.sin((wind_direction - 270.0) * np.pi / 180.0) * layout_y
    )
    turbines_ordered = np.argsort(layout_x_rot)

    # Remove turbines that need not be optimized
    num_turbs = len(layout_x)
    if turbs_to_opt is None:
        turbs_to_opt = range(num_turbs)
    turbines_ordered = [ti for ti in turbines_ordered if ti in turbs_to_opt]

    # Initialize optimal solution
    J_farm_opt = -1

    if turbine_weights is None:
        turbine_weights = np.ones_like(layout_x)

    yaw_angles_opt = np.array(yaw_angles_template, dtype=float)
    for ti in turbines_ordered:
        yaw_opt = yaw_angles_template[ti]  # Initialize
        for yaw in yaw_grid[ti]:
            if (not (yaw == yaw_angles_template[ti]) & ti > 0):  # Exclude case that we already evaluated
                yaw_angles_opt[ti] = yaw
                fi.calculate_wake(yaw_angles=yaw_angles_opt)
                turbine_powers = fi.get_turbine_power(
                    include_unc=include_unc,
                    unc_pmfs=unc_pmfs,
                    unc_options=unc_options,
                )
                test_power = np.dot(turbine_weights, turbine_powers)
                if test_power > J_farm_opt:
                    J_farm_opt = test_power
                    yaw_opt = yaw
        yaw_angles_opt[ti] = yaw_opt

    return yaw_angles_opt


def minimize_sr(fi, yaw_angles_template, bounds, opt_options=None,
                include_unc=False, unc_pmfs=None, unc_options=None,
                turbine_weights=None, turbs_to_opt=None):
    # Load default settings, if unspecified
    if opt_options is None:
        opt_options = dict(
            {
                "Ny_passes": [5, 5],
                # "refine_solution": False,
                # "refine_method": "SLSQP",
                # "refine_options": {
                #     'maxiter': 10,
                #     'disp': False,
                #     'iprint': 1,
                #     'ftol': 1e-7,
                #     'eps': 0.01
                # }
            }
        )

    num_turbs = len(fi.layout_x)
    if turbine_weights is None:
        turbine_weights = np.ones(num_turbs, dtype=float)

    if turbs_to_opt is None:
        turbs_to_opt = range(num_turbs)

    # Confirm Ny_firstpass and Ny_secondpass are odd integers
    for Ny in opt_options["Ny_passes"]:
        if (not isinstance(Ny, int)) or (Ny % 2 == 0):
            raise ValueError("Ny_passes must contain exclusively odd integers")

    # Initialization
    yaw_search_space = [[]] * num_turbs  # Initialize as empty
    yaw_grid_offsets = [np.mean(b) for b in bounds]
    bounds_relative = [[b[0]-y, b[1]-y] for b, y in zip(bounds, yaw_grid_offsets)]

    # Perform each pass with Ny yaw angles
    for ii, Ny in enumerate(opt_options["Ny_passes"]):
        for ti in range(num_turbs):
            lb = bounds_relative[ti][0]
            if (lb + yaw_grid_offsets[ti]) < bounds[ti][0]:
                lb = (bounds[ti][0] - yaw_grid_offsets[ti])
    
            ub = bounds_relative[ti][1]
            if (ub + yaw_grid_offsets[ti]) > bounds[ti][1]:
                ub = (bounds[ti][1] - yaw_grid_offsets[ti])

            yaw_search_space[ti] = (
                yaw_grid_offsets[ti] + np.linspace(lb, ub, Ny)
            )

        # Find optimal solution
        yaw_angles_opt = _serial_refine_single_pass(
            fi=fi,
            yaw_grid=yaw_search_space,
            yaw_angles_template=yaw_angles_template,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            turbine_weights=turbine_weights
        )

        # Update variables if to do more passes
        if (ii < len(opt_options["Ny_passes"]) - 1):
            yaw_angles_template = yaw_angles_opt  # Overwrite initial cond.
            yaw_grid_offsets = yaw_angles_opt  # Refine search space
            for ti in range(num_turbs):
                dx = float(np.diff(bounds_relative[ti]) / (Ny - 1))
                bounds_relative[ti] = [-dx/2.0, dx/2.0]

    return yaw_angles_opt


# if __name__ == "__main__":
#     import os
#     import numpy as np
#     import floris.tools as wfct

#     file_dir = os.path.dirname(os.path.abspath(__file__))
#     fi = wfct.floris_interface.FlorisInterface(
#         "/home/bartdoekemeijer/python_scripts/floris/examples//example_input.json"
#     )

#     # Set turbine locations to 3 turbines in a row
#     D = fi.floris.farm.turbines[0].rotor_diameter
#     layout_x = [0, 7 * D, 14 * D, 21 * D]
#     layout_y = [0, 0, 0, 0.]
#     fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

#     for wd in [0.]: #np.arange(0., 360., 2.):
#         print('wd:', wd)
#         fi.reinitialize_flow_field(wind_direction=270.)

#         # Instantiate the Optimization object
#         yaw_angles = minimize(
#             fi=fi,
#             yaw_angles_template=np.zeros(4),
#             bounds=[[0.0, 20.0]] * 4
#         )

#     print('Yaw angles: ', yaw_angles)
#     # Solution: [20.  20.  17.5  0. ]