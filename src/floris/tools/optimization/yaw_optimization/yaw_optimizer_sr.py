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
        wd_array,
        ws_array,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5, 4],  # Optimization options
        turbine_weights=None,
        exclude_downstream_turbines=True,
        reduce_ngrid=False,
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
            wd_array=wd_array,
            ws_array=ws_array,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
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

        # Set baseline and optimization settings
        if reduce_ngrid:
            for ti in range(self.nturbs):
                # Force number of grid points to 2
                self.fi.floris.farm.turbines[ti].ngrid = 2
                self.fi.floris.farm.turbines[ti].initialize_turbine()
                print("Reducing ngrid. Unsure if this functionality works!")

        # Save optimization choices to self
        self.Ny_passes = Ny_passes

        # For each wind direction, determine the order of turbines
        self._get_turbine_orders()

        # Initialize optimum yaw angles and cost function as baseline values
        self._yaw_angles_opt = copy.deepcopy(self.yaw_angles_baseline)
        self._farm_power_opt = copy.deepcopy(self.farm_power_baseline)
        self._yaw_lbs = copy.deepcopy(self.minimum_yaw_angle)
        self._yaw_ubs = copy.deepcopy(self.maximum_yaw_angle)

    def _get_turbine_orders(self):
        layout_x = self.fi.layout_x
        layout_y = self.fi.layout_y
        turbines_ordered_array = []
        for wd in self.wd_array:
            layout_x_rot = (
                np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
                - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
            )
            turbines_ordered = np.argsort(layout_x_rot)
            turbines_ordered_array.append(turbines_ordered)
        self.turbines_ordered_array = turbines_ordered_array

    def _calc_powers_with_memory(self, yaw_angles, use_memory=True):
        # Initialize empty matrix
        farm_powers = np.zeros(yaw_angles.shape[0])

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            idx = (np.abs(self._yaw_angles_opt - yaw_angles) < 0.01).all(axis=1)
            farm_powers[idx] = self._farm_power_opt[idx]
            # print("Skipping {:d}/{:d} calculations: already in memory.".format(np.sum(idx), len(idx)))
        else:
            idx = np.array([False] * yaw_angles.shape[0], dtype=bool)

        # Now calculate farm powers for conditions we haven't yet evaluated previously
        start_time = timerpc()
        farm_powers[~idx] = self._calculate_farm_power(
            wd_array=self.wd_array[~idx],
            ws_array=self.ws_array[~idx],
            turbine_weights=self.turbine_weights[~idx, :],
            yaw_angles=yaw_angles[~idx, :],
        )
        self.time_spent_in_floris += (timerpc() - start_time)
        return farm_powers

    def _print_uplift(self):
        pow_opt = self._farm_power_opt
        pow_bl = self.farm_power_baseline
        print("Windrose farm power uplift = {:.2f} %".format(100 * (np.sum(pow_opt) / np.sum(pow_bl) - 1)))
        print("Farm power uplift per direction:")
        E = 100.0 * (pow_opt / pow_bl - 1)
        for iw in range(len(self.wd_array)):
            print("  WD: {:.1f} deg, WS: {:.1f} m/s -- uplift: {:.3f} %".format(self.wd_array[iw], self.ws_array[iw], E[iw]))

    def _generate_evaluation_grid(self, pass_depth, turbine_depth):
        """
        Calculate the yaw angles for every iteration in the SR algorithm, for turbine,
        for every wind direction, for every wind speed, for every TI. Basically, this
        should yield a grid of yaw angle sets to evaluate the wind farm AEP with 'Ny'
        times. Then, for each ambient condition set, 
        """

        # Initialize yaw angles to evaluate, 'Ny' times the wind rose
        Ny = self.Ny_passes[pass_depth]
        evaluation_grid = np.tile(self._yaw_angles_opt, (Ny, 1, 1))

        # Get a list of the turbines in order of x and sort front to back
        for iw in range(len(self.wd_array)):
            turbid = self.turbines_ordered_array[iw][turbine_depth]  # Turbine to manipulate

            # Check if this turbine needs to be optimized. If not, continue
            if not self.turbs_to_opt[iw, turbid]:
                continue

            # # Remove turbines that need not be optimized
            # turbines_ordered = [ti for ti in turbines_ordered if ti in self.turbs_to_opt]

            # Grab yaw bounds from self
            yaw_lb = self._yaw_lbs[iw, turbid]
            yaw_ub = self._yaw_ubs[iw, turbid]

            # Saturate to allowable yaw limits
            if yaw_lb < self.minimum_yaw_angle[iw, turbid]:
                yaw_lb = self.minimum_yaw_angle[iw, turbid]
            if yaw_ub > self.maximum_yaw_angle[iw, turbid]:
                yaw_ub = self.maximum_yaw_angle[iw, turbid]

            if pass_depth == 0:
                # Evaluate all possible coordinates
                yaw_angles = np.linspace(yaw_lb, yaw_ub, Ny)
            else:
                # Remove middle point: was evaluated in previous iteration
                c = int(Ny / 2)  # Central point (to remove)
                ids = [*list(range(0, c)), *list(range(c + 1, Ny + 1))]
                yaw_angles = np.linspace(yaw_lb, yaw_ub, Ny + 1)[ids]

            for iii in range(Ny):
                evaluation_grid[iii, iw, turbid] = yaw_angles[iii]

        self._yaw_evaluation_grid = evaluation_grid
        return evaluation_grid

    def _process_evaluation_grid(self):
        # Evaluate the farm AEPs for the grid of possible yaw angles
        evaluation_grid = self._yaw_evaluation_grid
        farm_powers = np.zeros_like(evaluation_grid)[:, :, 0]
        for iii in range(evaluation_grid.shape[0]):
            farm_powers[iii, :] = self._calc_powers_with_memory(evaluation_grid[iii, :, :])
        return farm_powers

    def _optimize(self): 
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """
        # Determine which turbines need not be optimized
        self._reduce_control_variables()

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
                for iw in range(len(self.wd_array)):
                    # Get id of the turbine of which we manipulated the yaw angle
                    turbid = self.turbines_ordered_array[iw][turbine_depth]
                    arg_opt = args_opt[iw]  # For this farm, arg_opt increases power the most

                    # Check if farm power increased compared to previous iteration
                    pow_opt = farm_powers[arg_opt, iw]
                    pow_prev_iteration = self._farm_power_opt[iw]
                    if pow_opt > pow_prev_iteration:
                        xopt = evaluation_grid[arg_opt, iw, turbid]
                        self._yaw_angles_opt[iw, turbid] = xopt  # Update optimal yaw angle
                        self._farm_power_opt[iw] = pow_opt  # Update optimal power
                    else:
                        # print("Power did not increase compared to previous evaluation")
                        xopt = self._yaw_angles_opt[iw, turbid]

                    # Update bounds for next iteration to close proximity of optimal solution
                    dx = evaluation_grid[1, iw, turbid] - evaluation_grid[0, iw, turbid]
                    lb_next = xopt - 0.50 * dx
                    ub_next = xopt + 0.50 * dx
                    if lb_next < self.minimum_yaw_angle[iw, turbid]:
                        lb_next = self.minimum_yaw_angle[iw, turbid]
                    if ub_next > self.maximum_yaw_angle[iw, turbid]:
                        ub_next = self.maximum_yaw_angle[iw, turbid]
                    self._yaw_lbs[iw, turbid] = lb_next
                    self._yaw_ubs[iw, turbid] = ub_next
                
                # self._print_uplift()

        # Gather optimal results
        self.yaw_angles_opt = np.array(self._yaw_angles_opt, copy=True)
        self.farm_power_opt = np.array(self._farm_power_opt, copy=True)

        # Produce output table
        ti = np.min(self.fi.floris.farm.turbulence_intensity)
        df_opt = pd.DataFrame({
            "wind_direction": self.wd_array,
            "wind_speed": self.ws_array,
            "turbulence_intensity": np.ones_like(self.wd_array) * ti,
            "yaw_angles_opt": [yaw_angles for yaw_angles in self._yaw_angles_opt],
            "farm_power_opt": self.farm_power_opt,
            "farm_power_baseline": self.farm_power_baseline,
        })
        return df_opt
