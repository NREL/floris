# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import copy
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

# from .yaw_optimizer_scipy import YawOptimizationScipy
from .yaw_optimization_base import YawOptimization


class YawOptimizationSR(YawOptimization):
    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5, 4],  # Optimization options
        turbine_weights=None,
        exclude_downstream_turbines=True,
        exploit_layout_symmetry=True,
        # reduce_ngrid=False,
        # cluster_turbines=False,
        # cluster_wake_slope=0.30,
        # verify_convergence=True,
    ):
        """
        Instantiate YawOptimizationSR object with a FlorisInterface object
        and assign parameter values.
        """

        # Initialize base class
        super().__init__(
            fi=fi,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            exploit_layout_symmetry=exploit_layout_symmetry,
            # cluster_turbines=cluster_turbines,
            # cluster_wake_slope=cluster_wake_slope,
            # verify_convergence=verify_convergence,
        )

        # Start a timer for FLORIS computations
        self.time_spent_in_floris = 0

        # Confirm that Ny_passes are integers and odd/even
        for Nii, Ny in enumerate(Ny_passes):
            if not isinstance(Ny, int):
                raise ValueError("Ny_passes must contain exclusively integers")
            if Ny < 2:
                raise ValueError("Each entry in Ny_passes must have a value of at least 2.")
            if (Nii > 0) & ((Ny + 1) % 2 == 0):
                raise ValueError("The second and further entries of Ny_passes must be even numbers. " + 
                    "This is to ensure the same yaw angles are not evaluated twice between passes.")

        # # Set baseline and optimization settings
        # if reduce_ngrid:
        #     for ti in range(self.nturbs):
        #         # Force number of grid points to 2
        #         self.fi.floris.farm.turbines[ti].ngrid = 2
        #         self.fi.floris.farm.turbines[ti].initialize_turbine()
        #         print("Reducing ngrid. Unsure if this functionality works!")

        # Save optimization choices to self
        self.Ny_passes = Ny_passes

        # For each wind direction, determine the order of turbines
        self._get_turbine_orders()

        # Initialize optimum yaw angles and cost function as baseline values
        self._yaw_angles_opt_subset = copy.deepcopy(self._yaw_angles_baseline_subset)
        self._farm_power_opt_subset = copy.deepcopy(self._farm_power_baseline_subset)
        self._yaw_lbs = copy.deepcopy(self._minimum_yaw_angle_subset)
        self._yaw_ubs = copy.deepcopy(self._maximum_yaw_angle_subset)

    def _get_turbine_orders(self):
        layout_x = self.fi.layout_x
        layout_y = self.fi.layout_y
        turbines_ordered_array = []
        for wd in self.fi_subset.floris.flow_field.wind_directions:
            layout_x_rot = (
                np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
                - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
            )
            turbines_ordered = np.argsort(layout_x_rot)
            turbines_ordered_array.append(turbines_ordered)
        self.turbines_ordered_array_subset = turbines_ordered_array


    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        # Define current optimal solutions and floris wind directions locally
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        farm_power_opt_subset = self._farm_power_opt_subset
        wd_array_subset = self.fi_subset.floris.flow_field.wind_directions
        turbine_weights_subset = self._turbine_weights_subset

        # Reformat yaw_angles_subset, if necessary
        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 4)
        if eval_multiple_passes:
            # Four-dimensional; format everything into three-dimensional
            Ny = yaw_angles_subset.shape[0]  # Number of passes
            yaw_angles_subset = np.vstack(
                [yaw_angles_subset[iii, :, :, :] for iii in range(Ny)]
            )
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1, 1))
            farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny, 1))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1, 1))

        # Initialize empty matrix for floris farm power outputs
        farm_powers = np.zeros((yaw_angles_subset.shape[0], yaw_angles_subset.shape[1]))

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=2).flatten()
            farm_powers[idx, 0] = farm_power_opt_subset[idx, 0]
            # print("Skipping {:d}/{:d} calculations: already in memory.".format(np.sum(idx), len(idx)))
        else:
            idx = np.array([False] * yaw_angles_subset.shape[0], dtype=bool)

        if not np.all(idx):
            # Now calculate farm powers for conditions we haven't yet evaluated previously
            start_time = timerpc()
            farm_powers[~idx] = self._calculate_farm_power(
                wd_array=wd_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :, :],
                yaw_angles=yaw_angles_subset[~idx, :, :],
            )
            self.time_spent_in_floris += (timerpc() - start_time)

        # Finally format solutions back to original format, if necessary
        if eval_multiple_passes:
            farm_powers = np.reshape(farm_powers, (Ny, -1, 1))

        return farm_powers

    # def _print_uplift(self):
    #     pow_opt = self._farm_power_opt
    #     pow_bl = self.farm_power_baseline
    #     print("Windrose farm power uplift = {:.2f} %".format(100 * (np.sum(pow_opt) / np.sum(pow_bl) - 1)))
    #     print("Farm power uplift per direction:")
    #     E = 100.0 * (pow_opt / pow_bl - 1)
    #     wd_array = self.fi.floris.flow_field.wind_directions
    #     ws = self.fi.floris.flow_field.wind_speeds[0]
    #     for iw in range(self.nconds):
    #         print("  WD: {:.1f} deg, WS: {:.1f} m/s -- uplift: {:.3f} %".format(wd_array[iw], ws, E[iw]))

    def _generate_evaluation_grid(self, pass_depth, turbine_depth):
        """
        Calculate the yaw angles for every iteration in the SR algorithm, for turbine,
        for every wind direction, for every wind speed, for every TI. Basically, this
        should yield a grid of yaw angle sets to evaluate the wind farm AEP with 'Ny'
        times. Then, for each ambient condition set, 
        """

        # Initialize yaw angles to evaluate, 'Ny' times the wind rose
        Ny = self.Ny_passes[pass_depth]
        evaluation_grid = np.tile(self._yaw_angles_opt_subset, (Ny, 1, 1, 1))

        # Get a list of the turbines in order of x and sort front to back
        for iw in range(self._nconds_subset):
            turbid = self.turbines_ordered_array_subset[iw][turbine_depth]  # Turbine to manipulate

            # Check if this turbine needs to be optimized. If not, continue
            if not self._turbs_to_opt_subset[iw, 0, turbid]:
                continue

            # # Remove turbines that need not be optimized
            # turbines_ordered = [ti for ti in turbines_ordered if ti in self.turbs_to_opt]

            # Grab yaw bounds from self
            yaw_lb = self._yaw_lbs[iw, 0, turbid]
            yaw_ub = self._yaw_ubs[iw, 0, turbid]

            # Saturate to allowable yaw limits
            if yaw_lb < self.minimum_yaw_angle[iw, 0, turbid]:
                yaw_lb = self.minimum_yaw_angle[iw, 0, turbid]
            if yaw_ub > self.maximum_yaw_angle[iw, 0, turbid]:
                yaw_ub = self.maximum_yaw_angle[iw, 0, turbid]

            if pass_depth == 0:
                # Evaluate all possible coordinates
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny)
            else:
                # Remove middle point: was evaluated in previous iteration
                c = int(Ny / 2)  # Central point (to remove)
                ids = [*list(range(0, c)), *list(range(c + 1, Ny + 1))]
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny + 1)[ids]

            for iii in range(Ny):
                evaluation_grid[iii, iw, 0, turbid] = yaw_angles_subset[iii]

        self._yaw_evaluation_grid = evaluation_grid
        return evaluation_grid

    def _process_evaluation_grid(self):
        # Evaluate the farm AEPs for the grid of possible yaw angles
        evaluation_grid = self._yaw_evaluation_grid
        farm_powers = self._calc_powers_with_memory(evaluation_grid)
        return farm_powers

    def _optimize(self): 
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """
        # For each pass, from front to back
        ii = 0
        for Nii in range(len(self.Ny_passes)):
            # Disturb yaw angles for one turbine at a time, from front to back
            for turbine_depth in range(self.nturbs):
                p = 100.0 * ii / (len(self.Ny_passes) * self.nturbs)
                ii += 1
                print("[Serial Refine] Processing pass={:d}, turbine_depth={:d} ({:.1f} %)".format(Nii, turbine_depth, p))

                # Create grid to evaluate yaw angles for one turbine == turbine_depth
                evaluation_grid = self._generate_evaluation_grid(pass_depth=Nii, turbine_depth=turbine_depth)

                # Evaluate grid of yaw angles, get farm powers and find optimal solutions
                farm_powers = self._process_evaluation_grid()
                args_opt = np.argmax(farm_powers, axis=0)

                # Update current optimal solution
                for iw in range(self._nconds_subset):
                    # Get id of the turbine of which we manipulated the yaw angle
                    turbid = self.turbines_ordered_array_subset[iw][turbine_depth]
                    arg_opt = args_opt[iw]  # For this farm, arg_opt increases power the most

                    # Check if farm power increased compared to previous iteration
                    pow_opt = farm_powers[arg_opt, iw]
                    pow_prev_iteration = self._farm_power_opt_subset[iw]
                    if pow_opt > pow_prev_iteration:
                        xopt = evaluation_grid[arg_opt, iw, 0, turbid]
                        self._yaw_angles_opt_subset[iw, 0, turbid] = xopt  # Update optimal yaw angle
                        self._farm_power_opt_subset[iw, 0] = pow_opt  # Update optimal power
                    else:
                        # print("Power did not increase compared to previous evaluation")
                        xopt = self._yaw_angles_opt_subset[iw, 0, turbid]

                    # Update bounds for next iteration to close proximity of optimal solution
                    dx = evaluation_grid[1, iw, 0, turbid] - evaluation_grid[0, iw, 0, turbid]
                    lb_next = xopt - 0.50 * dx
                    ub_next = xopt + 0.50 * dx
                    if lb_next < self.minimum_yaw_angle[iw, 0, turbid]:
                        lb_next = self.minimum_yaw_angle[iw, 0, turbid]
                    if ub_next > self.maximum_yaw_angle[iw, 0, turbid]:
                        ub_next = self.maximum_yaw_angle[iw, 0, turbid]
                    self._yaw_lbs[iw, 0, turbid] = lb_next
                    self._yaw_ubs[iw, 0, turbid] = ub_next
                
                # self._print_uplift()

        # Finalize optimization, i.e., retrieve full solutions
        self._finalize()

        # Produce output table
        ti = np.min(self.fi.floris.flow_field.turbulence_intensity)
        df_list = []
        for ii, wind_speed in enumerate(self.fi.floris.flow_field.wind_speeds):
            df_list.append(pd.DataFrame({
                "wind_direction": self.fi.floris.flow_field.wind_directions,
                "wind_speed": np.ones(self.nconds) * wind_speed,
                "turbulence_intensity": np.ones(self.nconds) * ti,
                "yaw_angles_opt": [yaw_angles for yaw_angles in self.yaw_angles_opt[:, ii, :]],
                "farm_power_opt": self.farm_power_opt[:, ii],
                "farm_power_baseline": self.farm_power_baseline[:, ii],
            }))
        df_opt = pd.concat(df_list, axis=0)
        return df_opt
