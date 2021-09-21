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


import numpy as np
from scipy.stats import norm

from .yaw_optimization_tools import cluster_turbines, derive_downstream_turbines


class YawOptimization:
    """
    YawOptimization is a subclass of :py:class:`floris.tools.optimization.scipy.
    Optimization` that is used to optimize the yaw angles of all turbines in a Floris
    Farm for a single set of inflow conditions using the SciPy optimize package.
    """

    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        calc_init_power=True,
        exclude_downstream_turbines=False,
        cluster_turbines=False,
        cluster_wake_slope=0.30,
    ):
        """
        Instantiate YawOptimization object with a FlorisInterface object
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
            calc_init_power (bool, optional): If True, calculates initial
                wind farm power for each set of wind conditions. Defaults to
                True.
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
        """

        self.fi = fi
        self.unc_pmfs = unc_pmfs

        if unc_options is None:
            self.unc_options = {
                "std_wd": 4.95,
                "std_yaw": 1.75,
                "pmf_res": 1.0,
                "pdf_cutoff": 0.995,
            }

        self.reinitialize_opt(
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            bnds=bnds,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            turbine_weights=turbine_weights,
            calc_init_power=calc_init_power,
            exclude_downstream_turbines=exclude_downstream_turbines,
            cluster_turbines=cluster_turbines,
            cluster_wake_slope=cluster_wake_slope,
            class_initialization=True
        )

    # Private methods

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in range(self.nturbs)]

    def _reduce_control_variables(self):
        """This function reduces the control problem by eliminating turbines
        of which the yaw angles need not be optimized, either because of a
        user-specified set of bounds (where bounds[i][0] == bounds[i][1]),
        or alternatively turbines that are far downstream in the wind farm
        and of which the wake does not impinge other turbines, if the
        boolean exclude_downstream_turbines == True. The normalized initial
        conditions and bounds are then calculated for the subset of turbines,
        to be used in the optimization.
        """
        if self.bnds is not None:
            self.turbs_to_opt, _ = np.where(np.abs(np.diff(self.bnds)) >= 0.001)
        else:
            self.turbs_to_opt = np.array(range(self.nturbs), dtype=int)

        if self.exclude_downstream_turbines:
            # Remove turbines from turbs_to_opt that are downstream
            downstream_turbines = derive_downstream_turbines(
                fi=self.fi, wind_direction=self.fi.floris.farm.wind_direction[0]
            )
            downstream_turbines = np.array(downstream_turbines, dtype=int)
            self.turbs_to_opt = [
                i for i in self.turbs_to_opt if i not in downstream_turbines
            ]

        # Set up a template yaw angles array with default solutions. The default
        # solutions are either 0.0 or the allowable yaw angle closest to 0.0 deg.
        # This solution addresses both downstream turbines, minimizing their abs.
        # yaw offset, and additionally fixing equality-constrained turbines to
        # their appropriate yaw angle.
        yaw_angles_template = np.zeros(self.nturbs, dtype=float)
        for ti in range(self.nturbs):
            if (self.bnds[ti][0] > 0.0) | (self.bnds[ti][1] < 0.0):
                yaw_angles_template[ti] = self.bnds[ti][
                    np.argmin(np.abs(self.bnds[ti]))
                ]
        self.yaw_angles_template = yaw_angles_template

        # Derive normalized initial condition and bounds
        self.x0_subset = [self.x0[i] for i in self.turbs_to_opt]

    def _normalize_control_variables(self):
        self.x0_norm = self._norm(
            np.array(self.x0_subset), self.minimum_yaw_angle, self.maximum_yaw_angle
        )
        self.bnds_norm = [
            (
                self._norm(
                    self.bnds[i][0], self.minimum_yaw_angle, self.maximum_yaw_angle
                ),
                self._norm(
                    self.bnds[i][1], self.minimum_yaw_angle, self.maximum_yaw_angle
                ),
            )
            for i in self.turbs_to_opt
        ]

    def _calculate_initial_farm_power(self):
        if self.calc_init_power:
            self.fi.calculate_wake(yaw_angles=self.yaw_angles_baseline)
            turbine_powers = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options,
            )
            self.initial_farm_power = np.dot(self.turbine_weights, turbine_powers)

    def _cluster_turbines(self):
        if not self.cluster_turbines:
            self.clusters = np.array([range(self.nturbs)], dtype=int)
        else:
            wind_directions = self.fi.floris.farm.wind_direction
            if (np.std(wind_directions) > 0.001):
                raise ValueError("Wind must be uniform for clustering.")

            # Calculate turbine clusters
            self.clusters = cluster_turbines(
                fi=self.fi,
                wind_direction=self.fi.floris.farm.wind_direction[0],
                wake_slope=self.cluster_wake_slope
            )

    # Public methods

    def reinitialize_opt(
        self,
        minimum_yaw_angle=None,
        maximum_yaw_angle=None,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        include_unc=None,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        calc_init_power=True,
        exclude_downstream_turbines=None,
        cluster_turbines=None,
        cluster_wake_slope=None,
        class_initialization=False
    ):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.

        Args:
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to None.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to None.
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
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to None.
            opt_options (dictionary, optional): Optimization options used by
                scipy.optimize.minize. Defaults to None.
            include_unc (bool, optional): Determines whether wind direction or
                yaw uncertainty are included. If True, uncertainty in wind
                direction and/or yaw position is included when determining
                wind farm power. Uncertainty is included by computing the
                mean wind farm power for a distribution of wind direction
                and yaw position deviations from the intended wind direction
                and yaw angles. Defaults to None.
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
            calc_init_power (bool, optional): If True, calculates initial
                wind farm power for each set of wind conditions. Defaults to
                None.
            exclude_downstream_turbines (bool, optional): If True,
                automatically finds and excludes turbines that are most
                downstream from the optimization problem. This significantly
                reduces computation time at no loss in performance. The yaw
                angles of these downstream turbines are fixed to 0.0 deg if
                the yaw bounds specified in self.bnds allow that, or otherwise
                are fixed to the lower or upper yaw bound, whichever is closer
                to 0.0. Defaults to None.
        """
        if cluster_turbines is not None:
            self.cluster_turbines = cluster_turbines
        if cluster_wake_slope is not None:
            self.cluster_wake_slope = cluster_wake_slope

        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if yaw_angles_baseline is not None:
            self.yaw_angles_baseline = yaw_angles_baseline
        else:
            self.yaw_angles_baseline = [
                turbine.yaw_angle
                for turbine in self.fi.floris.farm.turbine_map.turbines
            ]
            if any(np.abs(self.yaw_angles_baseline) > 0.0):
                print(
                    "INFO: Baseline yaw angles were not specified and were derived from the floris object."
                )
                print(
                    "INFO: The inherent yaw angles in the floris object are not all 0.0 degrees."
                )

        if bnds is not None:
            self.bnds = bnds
            self.minimum_yaw_angle = np.min([bnds[i][0] for i in range(self.nturbs)])
            self.maximum_yaw_angle = np.max([bnds[i][1] for i in range(self.nturbs)])
        else:
            if class_initialization:
                self._set_opt_bounds(self.minimum_yaw_angle, self.maximum_yaw_angle)

        if x0 is not None:
            self.x0 = x0
        else:
            if class_initialization:
                self.x0 = np.zeros(self.nturbs, dtype=float)
                for ti in range(self.nturbs):
                    if (self.bnds[ti][0] > 0.0) | (self.bnds[ti][1] < 0.0):
                        self.x0[ti] = np.mean(self.bnds[ti])

        if any(
            np.array(self.yaw_angles_baseline) < np.array([b[0] for b in self.bnds])
        ):
            print("INFO: yaw_angles_baseline exceed lower bound constraints.")
        if any(
            np.array(self.yaw_angles_baseline) > np.array([b[1] for b in self.bnds])
        ):
            print("INFO: yaw_angles_baseline in FLORIS exceed upper bound constraints.")
        if any(np.array(self.x0) < np.array([b[0] for b in self.bnds])):
            raise ValueError("Initial guess x0 exceeds lower bound constraints.")
        if any(np.array(self.x0) > np.array([b[1] for b in self.bnds])):
            raise ValueError("Initial guess x0 exceeds upper bound constraints.")

        if include_unc is not None:
            self.include_unc = include_unc
        if unc_pmfs is not None:
            self.unc_pmfs = unc_pmfs
        if unc_options is not None:
            self.unc_options = unc_options

        if self.include_unc & (self.unc_pmfs is None):
            if self.unc_options is None:
                self.unc_options = {
                    "std_wd": 4.95,
                    "std_yaw": 1.75,
                    "pmf_res": 1.0,
                    "pdf_cutoff": 0.995,
                }

            # create normally distributed wd and yaw uncertainty pmfs
            if self.unc_options["std_wd"] > 0:
                wd_bnd = int(
                    np.ceil(
                        norm.ppf(
                            self.unc_options["pdf_cutoff"],
                            scale=self.unc_options["std_wd"],
                        )
                        / self.unc_options["pmf_res"]
                    )
                )
                wd_unc = np.linspace(
                    -1 * wd_bnd * self.unc_options["pmf_res"],
                    wd_bnd * self.unc_options["pmf_res"],
                    2 * wd_bnd + 1,
                )
                wd_unc_pmf = norm.pdf(wd_unc, scale=self.unc_options["std_wd"])
                # normalize so sum = 1.0
                wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf)
            else:
                wd_unc = np.zeros(1)
                wd_unc_pmf = np.ones(1)

            if self.unc_options["std_yaw"] > 0:
                yaw_bnd = int(
                    np.ceil(
                        norm.ppf(
                            self.unc_options["pdf_cutoff"],
                            scale=self.unc_options["std_yaw"],
                        )
                        / self.unc_options["pmf_res"]
                    )
                )
                yaw_unc = np.linspace(
                    -1 * yaw_bnd * self.unc_options["pmf_res"],
                    yaw_bnd * self.unc_options["pmf_res"],
                    2 * yaw_bnd + 1,
                )
                yaw_unc_pmf = norm.pdf(yaw_unc, scale=self.unc_options["std_yaw"])
                # normalize so sum = 1.0
                yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf)
            else:
                yaw_unc = np.zeros(1)
                yaw_unc_pmf = np.ones(1)

            self.unc_pmfs = {
                "wd_unc": wd_unc,
                "wd_unc_pmf": wd_unc_pmf,
                "yaw_unc": yaw_unc,
                "yaw_unc_pmf": yaw_unc_pmf,
            }

        if turbine_weights is None:
            self.turbine_weights = np.ones(self.nturbs)
        else:
            self.turbine_weights = np.array(turbine_weights, dtype=float)

        if calc_init_power is not None:
            self.calc_init_power = calc_init_power

        if exclude_downstream_turbines is not None:
            self.exclude_downstream_turbines = exclude_downstream_turbines
        
        self._calculate_initial_farm_power()
        self._cluster_turbines()

    def reinitialize_flow_field(self, wind_speed=None, wind_layout=None,
        wind_direction=None, wind_shear=None, wind_veer=None,
        specified_wind_height=None, turbulence_intensity=None,
        turbulence_kinetic_energy=None, air_density=None, wake=None,
        layout_array=None, with_resolution=None):

        self.fi.reinitialize_flow_field(
            wind_speed=wind_speed,
            wind_layout=wind_layout,
            wind_direction=wind_direction,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            specified_wind_height=specified_wind_height,
            turbulence_intensity=turbulence_intensity,
            turbulence_kinetic_energy=turbulence_kinetic_energy,
            air_density=air_density,
            wake=wake,
            layout_array=layout_array,
            with_resolution=with_resolution,
        )
        self._calculate_initial_farm_power()
        self._cluster_turbines()

    def plot_clusters(self):
        if not self.cluster_turbines:
            print("Turbines are not clustered (cluster_turbines=False)")
        else:
            self.clusters = cluster_turbines(
                fi=self.fi,
                wind_direction=self.fi.floris.farm.wind_direction[0],
                wake_slope=self.cluster_wake_slope,
                plot_lines=True
            )

    # Supporting functions
    def _norm(self, val, x1, x2):
        return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    # Properties

    @property
    def minimum_yaw_angle(self):
        """
        The minimum yaw angle for the optimization. The setting-method
        updates the optimization bounds accordingly.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): The minimum yaw angle to set (deg).

        Returns:
            float: The minimum yaw angle currently set (deg).
        """
        return self._minimum_yaw_angle

    @minimum_yaw_angle.setter
    def minimum_yaw_angle(self, value):
        self._minimum_yaw_angle = value

    @property
    def maximum_yaw_angle(self):
        """
        The maximum yaw angle for the optimization. The setting-method
        updates the optimization bounds accordingly.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): The maximum yaw angle to set (deg).

        Returns:
            float: The maximum yaw angle currently set (deg).
        """
        return self._maximum_yaw_angle

    @maximum_yaw_angle.setter
    def maximum_yaw_angle(self, value):
        self._maximum_yaw_angle = value

    @property
    def x0(self):
        """
        The initial yaw angles used for the optimization.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (iterable): The yaw angle initial conditions to set (deg).

        Returns:
            list: The yaw angle initial conditions currently set (deg).
        """
        return self._x0

    @property
    def nturbs(self):
        """
        Number of turbines in the :py:class:`~.farm.Farm` object.

        Returns:
            int
        """
        self._nturbs = len(self.fi.floris.farm.turbine_map.turbines)
        return self._nturbs

    @x0.setter
    def x0(self, value):
        self._x0 = value
