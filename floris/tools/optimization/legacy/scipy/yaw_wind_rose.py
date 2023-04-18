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
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .derive_downstream_turbines import derive_downstream_turbines
from .optimization import Optimization


class YawOptimizationWindRose(Optimization):
    """
    YawOptimizationWindRose is a subclass of
    :py:class:`~.tools.optimization.scipy.Optimization` that is used to
    optimize the yaw angles of all turbines in a Floris Farm for multiple sets
    of inflow conditions (combinations of wind speed, wind direction, and
    optionally turbulence intensity) using the scipy optimize package.
    """

    def __init__(
        self,
        fi,
        wd,
        ws,
        ti=None,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        minimum_ws=3.0,
        maximum_ws=25.0,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        opt_method="SLSQP",
        opt_options=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        verbose=False,
        calc_init_power=True,
        exclude_downstream_turbines=False,
    ):
        """
        Instantiate YawOptimizationWindRose object with a FlorisInterface
        object and assign parameter values.

        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
            wd (iterable) : The wind directions for which the yaw angles are
                optimized (deg).
            ws (iterable): The wind speeds for which the yaw angles are
                optimized (m/s).
            ti (iterable, optional): An optional list of turbulence intensity
                values for which the yaw angles are optimized. If not
                specified, the current TI value in the Floris object will be
                used for all optimizations. Defaults to None.
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to 0.0.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to 25.0.
            minimum_ws (float, optional): Minimum wind speed at which
                optimization is performed (m/s). Assumes zero power generated
                below this value. Defaults to 3.
            maximum_ws (float, optional): Maximum wind speed at which
                optimization is performed (m/s). Assumes optimal yaw offsets
                are zero above this wind speed. Defaults to 25.
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
                scipy.optimize.minize. Defaults to 'SLSQP'.
            opt_options (dictionary, optional): Optimization options used by
                scipy.optimize.minize. If none are specified, they are set to
                {'maxiter': 100, 'disp': False, 'iprint': 1, 'ftol': 1e-7,
                'eps': 0.01}. Defaults to None.
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
        """
        super().__init__(fi)

        if opt_options is None:
            self.opt_options = {
                "maxiter": 100,
                "disp": False,
                "iprint": 1,
                "ftol": 1e-7,
                "eps": 0.01,
            }

        self.unc_pmfs = unc_pmfs

        if unc_options is None:
            self.unc_options = {
                "std_wd": 4.95,
                "std_yaw": 1.75,
                "pmf_res": 1.0,
                "pdf_cutoff": 0.995,
            }

        self.ti = ti

        self.reinitialize_opt_wind_rose(
            wd=wd,
            ws=ws,
            ti=ti,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            minimum_ws=minimum_ws,
            maximum_ws=maximum_ws,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            bnds=bnds,
            opt_method=opt_method,
            opt_options=opt_options,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            turbine_weights=turbine_weights,
            calc_init_power=calc_init_power,
            exclude_downstream_turbines=exclude_downstream_turbines,
        )

        self.verbose = verbose

    # Private methods

    def _get_initial_farm_power(self):
        self.initial_farm_powers = []

        for i in range(len(self.wd)):
            if (self.ws[i] >= self.minimum_ws) & (self.ws[i] <= self.maximum_ws):
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )

                # initial power
                self.fi.calculate_wake(yaw_angles=self.yaw_angles_baseline)
                power_init = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )
            elif self.ws[i] >= self.maximum_ws:
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )
                self.fi.calculate_wake(yaw_angles=self.yaw_angles_baseline)
                power_init = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )
            else:
                power_init = self.nturbs * [0.0]

            self.initial_farm_powers.append(np.dot(self.turbine_weights, power_init))

    def _get_power_for_yaw_angle_opt(self, yaw_angles_subset_norm):
        """
        Assign yaw angles to turbines, calculate wake, report power

        Args:
            yaw_angles_subset_norm (np.array): Yaw to apply to subset
            of controlled turbines, normalized.

        Returns:
            power (float): Wind plant power. #TODO negative? in kW?
        """
        yaw_angles_subset = self._unnorm(
            np.array(yaw_angles_subset_norm),
            self.minimum_yaw_angle,
            self.maximum_yaw_angle,
        )

        # Create a full yaw angle array
        yaw_angles = np.array(self.yaw_angles_template, copy=True)
        yaw_angles[self.turbs_to_opt] = yaw_angles_subset

        self.fi.calculate_wake(yaw_angles=yaw_angles)
        turbine_powers = self.fi.get_turbine_power(
            include_unc=self.include_unc,
            unc_pmfs=self.unc_pmfs,
            unc_options=self.unc_options,
        )

        return (
            -1.0
            * np.dot(self.turbine_weights, turbine_powers)
            / self.initial_farm_power
        )

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        """
        Sets minimum and maximum yaw angle bounds for optimization.
        """

        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in range(self.nturbs)]

    def _optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditions (wind speed, direction, etc.).

        Returns:
            opt_yaw_angles (np.array): Optimal yaw angles of each turbine.
        """
        opt_yaw_angles = np.array(self.yaw_angles_template, copy=True)
        wind_map = self.fi.floris.farm.wind_map
        self._reduce_control_variables()

        if len(self.turbs_to_opt) > 0:
            self.residual_plant = minimize(
                self._get_power_for_yaw_angle_opt,
                self.x0_norm,
                method=self.opt_method,
                bounds=self.bnds_norm,
                options=self.opt_options,
            )
            opt_yaw_angles_subset = self._unnorm(
                self.residual_plant.x, self.minimum_yaw_angle, self.maximum_yaw_angle
            )
            opt_yaw_angles[self.turbs_to_opt] = opt_yaw_angles_subset

        self.fi.reinitialize_flow_field(
            wind_speed=wind_map.input_speed,
            wind_direction=wind_map.input_direction,
            turbulence_intensity=wind_map.input_ti,
        )
        return opt_yaw_angles

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
        x0_subset = [self.x0[i] for i in self.turbs_to_opt]
        self.x0_norm = self._norm(
            np.array(x0_subset), self.minimum_yaw_angle, self.maximum_yaw_angle
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

    # Public methods

    def reinitialize_opt_wind_rose(
        self,
        wd=None,
        ws=None,
        ti=None,
        minimum_yaw_angle=None,
        maximum_yaw_angle=None,
        minimum_ws=None,
        maximum_ws=None,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        opt_method=None,
        opt_options=None,
        include_unc=None,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        calc_init_power=True,
        exclude_downstream_turbines=None,
    ):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.

        Args:
            wd (iterable, optional) : The wind directions for which the yaw
                angles are optimized (deg). Defaults to None.
            ws (iterable, optional): The wind speeds for which the yaw angles
                are optimized (m/s). Defaults to None.
            ti (iterable, optional): An optional list of turbulence intensity
                values for which the yaw angles are optimized. Defaults to None.
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to None.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to None.
            minimum_ws (float, optional): Minimum wind speed at which
                optimization is performed (m/s). Assumes zero power generated
                below this value. Defaults to None.
            maximum_ws (float, optional): Maximum wind speed at which
                optimization is performed (m/s). Assumes optimal yaw offsets
                are zero above this wind speed. Defaults to None.
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

        if wd is not None:
            self.wd = wd
        if ws is not None:
            self.ws = ws
        if ti is not None:
            self.ti = ti
        if minimum_ws is not None:
            self.minimum_ws = minimum_ws
        if maximum_ws is not None:
            self.maximum_ws = maximum_ws
        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options
        if yaw_angles_baseline is not None:
            self.yaw_angles_baseline = yaw_angles_baseline
        else:
            self.yaw_angles_baseline = [
                turbine.yaw_angle
                for turbine in self.fi.floris.farm.turbine_map.turbines
            ]
            if any(np.abs(self.yaw_angles_baseline) > 0.0):
                print(
                    "INFO: Baseline yaw angles were not specified and were derived "
                    "from the floris object."
                )
                print(
                    "INFO: The inherent yaw angles in the floris object are not all 0.0 degrees."
                )

        self.bnds = bnds
        if bnds is not None:
            self.minimum_yaw_angle = np.min([bnds[i][0] for i in range(self.nturbs)])
            self.maximum_yaw_angle = np.max([bnds[i][1] for i in range(self.nturbs)])
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, self.maximum_yaw_angle)

        if x0 is not None:
            self.x0 = x0
        else:
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

            # create normally distributed wd and yaw uncertaitny pmfs
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

        if calc_init_power:
            self._get_initial_farm_power()

        if exclude_downstream_turbines is not None:
            self.exclude_downstream_turbines = exclude_downstream_turbines
        self._reduce_control_variables()

    def calc_baseline_power(self):
        """
        This method computes the baseline power produced by the wind farm and
        the ideal power without wake losses for a series of wind speed, wind
        direction, and optionally TI combinations.

        Returns:
            pandas.DataFrame: A pandas DataFrame with the same number of rows
            as the length of the wd and ws arrays, containing the following
            columns:

                - **ws** (*float*) - The wind speed values for which power is
                computed (m/s).
                - **wd** (*float*) - The wind direction value for which power
                is calculated (deg).
                - **ti** (*float*) - The turbulence intensity value for which
                power is calculated. Only included if self.ti is not None.
                - **power_baseline** (*float*) - The total power produced by
                the wind farm with baseline yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced
                by the wind farm without wake losses (W).
                - **turbine_power_baseline** (*list* (*float*)) - A
                list containing the baseline power without wake steering for
                each wind turbine in the wind farm (W).
                - **turbine_power_no_wake** (*list* (*float*)) - A list
                containing the ideal power without wake losses for each wind
                turbine in the wind farm (W).
        """
        print("=====================================================")
        print("Calculating baseline power...")
        print("Number of wind conditions to calculate = ", len(self.wd))
        print("=====================================================")

        # Put results in dict for speed, instead of previously
        # appending to frame.
        result_dict = {}

        for i in range(len(self.wd)):
            if self.verbose:
                if self.ti is None:
                    print(
                        "Computing wind speed, wind direction pair "
                        + str(i)
                        + " out of "
                        + str(len(self.wd))
                        + ": wind speed = "
                        + str(self.ws[i])
                        + " m/s, wind direction = "
                        + str(self.wd[i])
                        + " deg."
                    )
                else:
                    print(
                        "Computing wind speed, wind direction, turbulence "
                        + "intensity set "
                        + str(i)
                        + " out of "
                        + str(len(self.wd))
                        + ": wind speed = "
                        + str(self.ws[i])
                        + " m/s, wind direction = "
                        + str(self.wd[i])
                        + " deg, turbulence intensity = "
                        + str(self.ti[i])
                        + "."
                    )

            # Find baseline power in FLORIS

            if self.ws[i] >= self.minimum_ws:
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )

                # calculate baseline power
                self.fi.calculate_wake(
                    yaw_angles=self.yaw_angles_baseline, no_wake=False
                )
                power_base = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )

                # calculate power for no wake case
                self.fi.calculate_wake(
                    yaw_angles=self.yaw_angles_baseline, no_wake=True
                )
                power_no_wake = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                    no_wake=True,
                )
            else:
                power_base = self.nturbs * [0.0]
                power_no_wake = self.nturbs * [0.0]

            # Include turbine weighing terms
            power_base = np.multiply(self.turbine_weights, power_base)
            power_no_wake = np.multiply(self.turbine_weights, power_no_wake)

            # add variables to dataframe
            if self.ti is None:
                result_dict[i] = {
                    "ws": self.ws[i],
                    "wd": self.wd[i],
                    "power_baseline": np.sum(power_base),
                    "turbine_power_baseline": power_base,
                    "power_no_wake": np.sum(power_no_wake),
                    "turbine_power_no_wake": power_no_wake,
                }
                # df_base = df_base.append(pd.DataFrame(
                #    {'ws':[self.ws[i]],'wd':[self.wd[i]],
                #    'power_baseline':[np.sum(power_base)],
                #    'turbine_power_baseline':[power_base],
                #    'power_no_wake':[np.sum(power_no_wake)],
                #    'turbine_power_no_wake':[power_no_wake]}))
            else:
                result_dict[i] = {
                    "ws": self.ws[i],
                    "wd": self.wd[i],
                    "ti": self.ti[i],
                    "power_baseline": np.sum(power_base),
                    "turbine_power_baseline": power_base,
                    "power_no_wake": np.sum(power_no_wake),
                    "turbine_power_no_wake": power_no_wake,
                }
                # df_base = df_base.append(pd.DataFrame(
                #    {'ws':[self.ws[i]],'wd':[self.wd[i]],
                #    'ti':[self.ti[i]],'power_baseline':[np.sum(power_base)],
                #    'turbine_power_baseline':[power_base],
                #    'power_no_wake':[np.sum(power_no_wake)],
                #    'turbine_power_no_wake':[power_no_wake]}))
        df_base = pd.DataFrame.from_dict(result_dict, "index")
        df_base.reset_index(drop=True, inplace=True)

        return df_base

    def optimize(self):
        """
        This method solves for the optimum turbine yaw angles for power
        production and the resulting power produced by the wind farm for a
        series of wind speed, wind direction, and optionally TI combinations.

        Returns:
            pandas.DataFrame: A pandas DataFrame with the same number of rows
            as the length of the wd and ws arrays, containing the following
            columns:

                - **ws** (*float*) - The wind speed values for which the yaw
                angles are optimized and power is computed (m/s).
                - **wd** (*float*) - The wind direction values for which the
                yaw angles are optimized and power is computed (deg).
                - **ti** (*float*) - The turbulence intensity values for which
                the yaw angles are optimized and power is computed. Only
                included if self.ti is not None.
                - **power_opt** (*float*) - The total power produced by the
                wind farm with optimal yaw offsets (W).
                - **turbine_power_opt** (*list* (*float*)) - A list
                containing the power produced by each wind turbine with optimal
                yaw offsets (W).
                - **yaw_angles** (*list* (*float*)) - A list containing
                the optimal yaw offsets for maximizing total wind farm power
                for each wind turbine (deg).
        """
        print("=====================================================")
        print("Optimizing wake redirection control...")
        print("Number of wind conditions to optimize = ", len(self.wd))
        print("Number of yaw angles to optimize = ", len(self.turbs_to_opt))
        print("=====================================================")

        df_opt = pd.DataFrame()

        for i in range(len(self.wd)):
            if self.verbose:
                if self.ti is None:
                    print(
                        "Computing wind speed, wind direction pair "
                        + str(i)
                        + " out of "
                        + str(len(self.wd))
                        + ": wind speed = "
                        + str(self.ws[i])
                        + " m/s, wind direction = "
                        + str(self.wd[i])
                        + " deg."
                    )
                else:
                    print(
                        "Computing wind speed, wind direction, turbulence "
                        + "intensity set "
                        + str(i)
                        + " out of "
                        + str(len(self.wd))
                        + ": wind speed = "
                        + str(self.ws[i])
                        + " m/s, wind direction = "
                        + str(self.wd[i])
                        + " deg, turbulence intensity = "
                        + str(self.ti[i])
                        + "."
                    )

            # Optimizing wake redirection control
            if (self.ws[i] >= self.minimum_ws) & (self.ws[i] <= self.maximum_ws):
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )

                self.initial_farm_power = self.initial_farm_powers[i]
                opt_yaw_angles = self._optimize()

                if np.sum(opt_yaw_angles) == 0:
                    print(
                        "No change in controls suggested for this inflow \
                        condition..."
                    )

                # optimized power
                self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
                power_opt = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )
            elif self.ws[i] >= self.maximum_ws:
                print(
                    "No change in controls suggested for this inflow \
                        condition..."
                )
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )
                opt_yaw_angles = np.array(self.yaw_angles_template, copy=True)
                self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
                power_opt = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )
            else:
                print(
                    "No change in controls suggested for this inflow \
                        condition..."
                )
                opt_yaw_angles = np.array(self.yaw_angles_template, copy=True)
                power_opt = self.nturbs * [0.0]

            # Include turbine weighing terms
            power_opt = np.multiply(self.turbine_weights, power_opt)

            # add variables to dataframe
            if self.ti is None:
                df_opt = df_opt.append(
                    pd.DataFrame(
                        {
                            "ws": [self.ws[i]],
                            "wd": [self.wd[i]],
                            "power_opt": [np.sum(power_opt)],
                            "turbine_power_opt": [power_opt],
                            "yaw_angles": [opt_yaw_angles],
                        }
                    )
                )
            else:
                df_opt = df_opt.append(
                    pd.DataFrame(
                        {
                            "ws": [self.ws[i]],
                            "wd": [self.wd[i]],
                            "ti": [self.ti[i]],
                            "power_opt": [np.sum(power_opt)],
                            "turbine_power_opt": [power_opt],
                            "yaw_angles": [opt_yaw_angles],
                        }
                    )
                )

        df_opt.reset_index(drop=True, inplace=True)

        return df_opt
