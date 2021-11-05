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


import copy

import numpy as np

from .yaw_optimizer_scipy import YawOptimizationScipy
from .yaw_optimization_base import YawOptimization


class YawOptimizationSR(YawOptimization):
    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        opt_options=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        exclude_downstream_turbines=True,
        cluster_turbines=False,
        cluster_wake_slope=0.30,
        verify_convergence=True,
    ):
        """
        Instantiate YawOptimizationScipy object with a FlorisInterface object
        and assign parameter values.

            Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to 0.0.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to 25.0.
            yaw_angles_baseline (iterable, optional): The baseline yaw
                angles used to calculate the initial and baseline power
                production in the wind farm and used to normalize the cost
                function. If none are specified, this variable is set equal
                to the current yaw angles in floris. Note that this variable
                need not meet the yaw constraints specified in self.bnds,
                yet a warning is raised if it does to inform the user.
                Defaults to None.
            x0 (iterable, optional): The initial guess for the optimization
                problem. These values must meet the constraints specified
                in self.bnds. Note that, if exclude_downstream_turbines=True,
                the initial guess for any downstream turbines are ignored
                since they are not part of the optimization. Instead, the yaw
                angles for those turbines are 0.0 if that meets the lower and
                upper bound, or otherwise as close to 0.0 as feasible. If no
                values for x0 are specified, x0 is set to be equal to zeros
                wherever feasible (w.r.t. the bounds), and equal to the
                average of its lower and upper bound for all non-downstream
                turbines otherwise. Defaults to None.
            bnds (iterable, optional): Bounds for the yaw angles, as tuples of
                min, max values for each turbine (deg). One can fix the yaw
                angle of certain turbines to a predefined value by setting that
                turbine's lower bound equal to its upper bound (i.e., an
                equality constraint), as: bnds[ti] = (x, x), where x is the
                fixed yaw angle assigned to the turbine. This works for both
                zero and nonzero yaw angles. Moreover, if
                exclude_downstream_turbines=True, the yaw angles for all
                downstream turbines will be 0.0 or a feasible value closest to
                0.0. If none are specified, the bounds are set to
                (minimum_yaw_angle, maximum_yaw_angle) for each turbine. Note
                that, if bnds is not none, its values overwrite any value given
                in minimum_yaw_angle and maximum_yaw_angle. Defaults to None.
            opt_options (dictionary, optional): Optimization options used by
                the serial refine method. If none are specified, will use the
                default option set: {"Ny_passes": [5, 5], "refine_solution":
                True, "refine_method": "SLSQP", "refine_options": {
                'maxiter': 10, 'disp': True, 'iprint': 1, 'ftol': 1e-7,
                'eps': 0.01}}. The variable Ny_passes is a list. If the list
                has two values, that means that the SR algorithm will do two
                passes, each refining the solution into X discrete points to
                evaluate the yaw angle at, with X being the value in the list.
                As the SR method provides an estimation of the optimal yaw
                angles at a discrete resolution, the user can opt to refine
                the solution using the SciPy optimization method. By default,
                solution refinement is enabled. Defaults to None.
            include_unc (bool, optional): Determines whether wind direction or
                yaw uncertainty are included. If True, uncertainty in wind
                direction and/or yaw position is included when determining
                wind farm power. Uncertainty is included by computing the
                mean wind farm power for a distribution of wind direction
                and yaw position deviations from the intended wind direction
                and yaw angles. Defaults to False.
            unc_pmfs (dictionary, optional): A dictionary containing
                probability mass functions describing the distribution of
                wind direction and yaw position deviations when wind direction
                and/or yaw position uncertainty is included in the power
                calculations. Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): The wind direction
                    deviations from the intended wind direction (deg).
                -   **wd_unc_pmf** (*np.array*): The probability
                    of each wind direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): The yaw angle deviations
                    from the intended yaw angles (deg).
                -   **yaw_unc_pmf** (*np.array*): The probability
                    of each yaw angle deviation in **yaw_unc** occuring.

                If none are specified, default PMFs are calculated using
                values provided in **unc_options**. Defaults to None.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): The standard deviation of
                    the wind direction deviations from the original wind
                    direction (deg).
                -   **std_yaw** (*float*): The standard deviation of
                    the yaw angle deviations from the original yaw angles (deg).
                -   **pmf_res** (*float*): The resolution in degrees
                    of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): The cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                If none are specified, default values of
                {'std_wd': 4.95, 'std_yaw': 1.75, 'pmf_res': 1.0,
                'pdf_cutoff': 0.995} are used. Defaults to None.
            turbine_weights (iterable, optional): weighing terms that allow
                the user to emphasize power gains at particular turbines or
                completely ignore power gains from other turbines. The array
                of turbine powers from floris is multiplied with this array
                in the calculation of the objective function. If None, this
                is an array with all values 1.0 and length equal to the
                number of turbines. Defaults to None.
            exclude_downstream_turbines (bool, optional): If True,
                automatically finds and excludes turbines that are most
                downstream from the optimization problem. This significantly
                reduces computation time at no loss in performance. The yaw
                angles of these downstream turbines are fixed to 0.0 deg if
                the yaw bounds specified in self.bnds allow that, or otherwise
                are fixed to the lower or upper yaw bound, whichever is closer
                to 0.0. Defaults to False.
            cluster_turbines (bool, optional): if True, clusters the wind
                turbines into sectors and optimizes the cost function for each
                farm sector separately. This can significantly reduce the
                computational cost involved if the farm can indeed be separated
                into multiple clusters. Defaults to False.
            cluster_wake_slope (float, optional): linear slope of the wake
                in the simplified linear expansion wake model (dy/dx). This
                model is used to derive wake interactions between turbines and
                to identify the turbine clusters. A good value is about equal
                to the turbulence intensity in FLORIS. Though, since yaw
                optimizations may shift the wake laterally, a safer option
                is twice the turbulence intensity. The default value is 0.30
                which should be valid for yaw optimizations at wd_std = 0.0 deg
                and turbulence intensities up to 15%. Defaults to 0.30.
            verify_convergence (bool, optional): specifies whether the found
                optimal yaw angles will be checked for accurately convergence.
                With large farms, especially when using SciPy or other global
                optimization methods, solutions do not always converge and
                turbines that should have a 0.0 deg actually have a 1.0 deg
                angle, for example. By enabling this function, the final yaw
                angles are compared to their baseline values one-by-one for
                the turbines to make sure no such convergence issues arise.
                Defaults to True.
        """
        if opt_options is None:
            # Default SR parameters
            opt_options = {
                "Ny_passes": [5, 5],
                "refine_solution": True,
                "refine_method": "SLSQP",
                "refine_options": {
                    "maxiter": 10,
                    "disp": True,
                    "iprint": 1,
                    "ftol": 1e-7,
                    "eps": 0.01,
                },
            }
        else:
            # Confirm Ny_firstpass and Ny_secondpass are odd integers
            for Ny in opt_options["Ny_passes"]:
                if (not isinstance(Ny, int)) or (Ny % 2 == 0):
                    raise ValueError("Ny_passes must contain exclusively odd integers")

        super().__init__(
            fi=fi,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            bnds=bnds,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            turbine_weights=turbine_weights,
            calc_init_power=False,
            exclude_downstream_turbines=exclude_downstream_turbines,
            cluster_turbines=cluster_turbines,
            cluster_wake_slope=cluster_wake_slope,
            verify_convergence=verify_convergence,
        )

        self.opt_options = opt_options

    def _serial_refine_single_pass(self, yaw_grid):
        """
        Perform a single pass of the serial refine optimization. The algorithm
        steps through all turbines in the wind farm, sorted according to their
        position w.r.t. the wind direction, upstream to downstream. The yaw
        angle of each turbine is evaluated for a predefined set of solutions
        and assigned the value which maximizes the power production.
        Fundamentally, this approaches the high-dimensional optimization
        problem by one single-dimensional optimization problem (1D grid
        search) for every turbine.

        Args:
            yaw_grid ([np.array]): Array of length equal to the number of
            turbines in the wind farm. Each entry contains a 1D array of
            length equal to Ny_passes[i], with `i` the pass number. The
            turbine will be evaluated at the values specified therein.

        Returns:
            yaw_angles_opt ([np.array]): Optimal yaw angles in degrees.
        """
        # Get a list of the turbines in order of x and sort front to back
        layout_x = self.fi.layout_x
        layout_y = self.fi.layout_y
        wind_direction = self.fi.floris.farm.wind_direction[0]
        layout_x_rot = (
            np.cos((wind_direction - 270.0) * np.pi / 180.0) * layout_x
            - np.sin((wind_direction - 270.0) * np.pi / 180.0) * layout_y
        )
        turbines_ordered = np.argsort(layout_x_rot)

        # Remove turbines that need not be optimized
        turbines_ordered = [ti for ti in turbines_ordered if ti in self.turbs_to_opt]

        J_farm_opt = -1  # Initialize optimal solution
        yaw_angles_opt = np.array(self.yaw_angles_template, dtype=float)
        for ti in turbines_ordered:
            yaw_opt = self.yaw_angles_template[ti]  # Initialize
            for yaw in yaw_grid[ti]:
                if (
                    not (yaw == self.yaw_angles_template[ti]) & ti > 0
                ):  # Exclude case that we already evaluated
                    yaw_angles_opt[ti] = yaw
                    self.fi.calculate_wake(yaw_angles=yaw_angles_opt)
                    turbine_powers = self.fi.get_turbine_power(
                        include_unc=self.include_unc,
                        unc_pmfs=self.unc_pmfs,
                        unc_options=self.unc_options,
                    )
                    test_power = np.dot(self.turbine_weights, turbine_powers)
                    if test_power > J_farm_opt:
                        J_farm_opt = test_power
                        yaw_opt = yaw
            yaw_angles_opt[ti] = yaw_opt

        return yaw_angles_opt

    def _refine_solution(self, x0):
        """
        This private function enables refinement of the found solution by
        the serial refinement method using a nonlinear optimization method
        such as SLSQP. This is useful since the serial refine method only
        evaluates yaw angles at discrete points at a resolution of 1.0 >
        degrees.

        Args:
            x0 ([np.array]): Initial conditions for the optimizer to start
            with. This should be the optimal yaw angles in degrees as found
            by the serial refine method.

        Returns:
            yaw_angles_opt ([np.array]): Optimal yaw angles in degrees.
        """
        refinement_solver = YawOptimizationScipy(
            fi=self.fi,
            minimum_yaw_angle=self.minimum_yaw_angle,
            maximum_yaw_angle=self.maximum_yaw_angle,
            yaw_angles_baseline=self.yaw_angles_baseline,
            x0=x0,
            bnds=self.bnds,
            opt_method=self.opt_options["refine_method"],
            opt_options=self.opt_options["refine_options"],
            include_unc=self.include_unc,
            unc_pmfs=self.unc_pmfs,
            unc_options=self.unc_options,
            turbine_weights=self.turbine_weights,
            exclude_downstream_turbines=self.exclude_downstream_turbines,
            cluster_turbines=False,
        )
        return refinement_solver.optimize()

    def _optimize(self):
        """
        Find optimum setting of turbine yaw angles for a single turbine
        cluster that maximizes the weighted wind farm power production
        given fixed atmospheric conditions (wind speed, direction, etc.)
        using the scipy.optimize.minimize function.

        Returns:
            opt_yaw_angles (np.array): Optimal yaw angles in degrees. This
            array is equal in length to the number of turbines in the farm.
        """
        # Reduce degrees of freedom and check if optimization necessary
        self._reduce_control_variables()
        if len(self.turbs_to_opt) <= 0:
            return self.yaw_angles_template

        # Initialization
        yaw_search_space = [[]] * self.nturbs  # Initialize as empty
        yaw_grid_offsets = [np.mean(b) for b in self.bnds]
        bounds_relative = [
            [b[0] - y, b[1] - y] for b, y in zip(self.bnds, yaw_grid_offsets)
        ]

        # Perform each pass with Ny yaw angles
        for ii, Ny in enumerate(self.opt_options["Ny_passes"]):
            for ti in range(self.nturbs):
                lb = bounds_relative[ti][0]
                if (lb + yaw_grid_offsets[ti]) < self.bnds[ti][0]:
                    lb = self.bnds[ti][0] - yaw_grid_offsets[ti]

                ub = bounds_relative[ti][1]
                if (ub + yaw_grid_offsets[ti]) > self.bnds[ti][1]:
                    ub = self.bnds[ti][1] - yaw_grid_offsets[ti]

                yaw_search_space[ti] = yaw_grid_offsets[ti] + np.linspace(lb, ub, Ny)

            # Single pass through all turbines
            yaw_angles_opt = self._serial_refine_single_pass(yaw_search_space)

            # Update variables if we are to do more passes
            if ii < len(self.opt_options["Ny_passes"]) - 1:
                # yaw_angles_template = yaw_angles_opt  # Overwrite initial cond.
                yaw_grid_offsets = yaw_angles_opt  # Refine search space
                for ti in range(self.nturbs):
                    dx = float(np.diff(bounds_relative[ti]) / (Ny - 1))
                    bounds_relative[ti] = [-dx / 2.0, dx / 2.0]

        if self.opt_options["refine_solution"]:
            # This refinement solver should already include verify_solution
            yaw_angles_opt = self._refine_solution(x0=yaw_angles_opt)
        else:
            # Without a refinement solver, do verify_solution ourselves
            yaw_angles_opt = self._verify_solution(yaw_angles_opt=yaw_angles_opt)

        return yaw_angles_opt

    def optimize(self):
        """
        Find optimum setting of turbine yaw angles for each of the turbine
        clusters that maximizes the weighted wind farm power production
        given fixed atmospheric conditions (wind speed, direction, etc.)
        using the scipy.optimize.minimize function.

        Returns:
            opt_yaw_angles (np.array): Optimal yaw angles in degrees. This
            array is equal in length to the number of turbines in the farm.
        """

        if not self.cluster_turbines:
            return self._optimize()

        else:
            xt = np.array(self.fi.layout_x, dtype=float)
            yt = np.array(self.fi.layout_y, dtype=float)
            opt_yaw_angles = np.zeros(self.nturbs, dtype=float)
            for cl in self.clusters:
                yopt_c = copy.deepcopy(self)
                yopt_c.yaw_angles_baseline = np.array(self.yaw_angles_baseline)[cl]
                yopt_c.turbine_weights = np.array(self.turbine_weights)[cl]
                yopt_c.bnds = np.array(self.bnds)[cl]
                yopt_c.x0 = np.array(self.x0)[cl]
                yopt_c.fi.reinitialize_flow_field(layout_array=[xt[cl], yt[cl]])
                opt_yaw_angles[cl] = yopt_c._optimize()

        return opt_yaw_angles
