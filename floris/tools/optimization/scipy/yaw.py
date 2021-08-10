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
from scipy.optimize import minimize

from floris.tools.optimization.scipy.derive_downstream_turbines import derive_downstream_turbines
from .optimization import Optimization


class YawOptimization(Optimization):
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
        x0=None,
        bnds=None,
        opt_method="SLSQP",
        opt_options=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        calc_init_power=True,
        exclude_downstream_turbines=False
    ):
        """
        Instantiate YawOptimization object with a FlorisInterface object
        and assign parameter values.

        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). Defaults to 0.0.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). Defaults to 25.0.
            x0 (iterable, optional): The initial yaw conditions (deg). If none
                are specified, they are set to the current yaw angles for
                all turbines. Defaults to None.
            bnds (iterable, optional): Bounds for the yaw angles (tuples of
                min, max values for each turbine (deg)). If none are
                specified, they are set to (minimum_yaw_angle,
                maximum_yaw_angle) for each turbine. Defaults to None.
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
            turbine_weights (iterable, optional): weighing terms that allows the
                user to emphasize power gains at particular turbines or
                completely ignore power gains from other turbines. The array
                of turbine powers from floris is multiplied with this array in
                the calculation of the objective function. Defaults to an array
                with all values 1.0 and length equal to the number of turbines.
            calc_init_power (bool, optional): If True, calculates initial wind
                farm power for each set of wind conditions. Defaults to True.
            exclude_downstream_turbines (bool, optional): If True, automatically
                finds and excludes turbines that are most downstream in the wind
                farm from the optimization problem. This can significantly reduce
                the computational time without losing any performance/potential
                AEP gain. Namely, downstream turbines are automatically fixed
                to 0.0 deg (or fixed to the closest bound to 0.0, if 0.0 is not
                possible).
        """
        super().__init__(fi)

        if opt_options is None:
            self.opt_options = {
                "maxiter": 50,
                "disp": True,
                "iprint": 2,
                "ftol": 1e-12,
                "eps": 0.1,
            }

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

    # Private methods

    def _yaw_power_opt(self, yaw_angles_subset_norm):
        # Unnorm subset
        yaw_angles_subset = self._unnorm(
            np.array(yaw_angles_subset_norm),
            self.minimum_yaw_angle,
            self.maximum_yaw_angle
        )
        # Create a full yaw angle array
        yaw_angles = np.array(self.x0, dtype=float)
        yaw_angles[self.turbs_to_opt] = yaw_angles_subset

        self.fi.calculate_wake(yaw_angles=yaw_angles)
        turbine_powers = self.fi.get_turbine_power(
            include_unc=self.include_unc,
            unc_pmfs=self.unc_pmfs,
            unc_options=self.unc_options,
        )

        return (
            -1. * np.dot(self.turbine_weights, turbine_powers)
            / self.initial_farm_power
        )

    def _optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        opt_yaw_angles = np.array(self.x0, dtype=float)
        self._reduce_control_variables()

        if len(self.turbs_to_opt) > 0:
            self.residual_plant = minimize(
                self._yaw_power_opt,
                self.x0_norm,
                method=self.opt_method,
                bounds=self.bnds_norm,
                options=self.opt_options,
            )

            opt_yaw_angles_subset = self._unnorm(
                self.residual_plant.x, self.minimum_yaw_angle, self.maximum_yaw_angle
            )
            opt_yaw_angles[self.turbs_to_opt] = opt_yaw_angles_subset

        return opt_yaw_angles

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in range(self.nturbs)]

    def _reduce_control_variables(self):
        """This function reduces the control problem by eliminating turbines
        of which the yaw angles need not be optimized, either because of a
        user-specified set of bounds (where bounds[i][0] == bounds[i][1]),
        or alternatively turbines that are far downstream in the wind farm
        and of which the wake does not impinge other turbines, if the
        boolean exclude_downstream_turbines == True.
        Secondly, the normalized initial conditions and bounds are calculated
        for the subset of turbines, to be then used in the optimization
        algorithm.
        """
        if self.bnds is not None:
            self.turbs_to_opt = np.where(np.diff(self.bnds) > 0.001)[0]
        else:
            self.turbs_to_opt = np.array(range(self.nturbs), dtype=int)

        if self.exclude_downstream_turbines:
            downstream_turbines = derive_downstream_turbines(
                fi=self.fi,
                wind_direction=self.fi.floris.farm.wind_direction[0]
            )
            for i in downstream_turbines:
                # Fix yaw angles to 0. or closest value for downstream turbines
                if i in self.turbs_to_opt:
                    if (
                        (self.bnds is None)
                        or
                        ((self.bnds[i][0] <= 0.) & (self.bnds[i][1] >= 0.))
                    ):
                        self.x0[i] = 0.0
                    else:
                        id_closest_to_zero = np.argmin(np.abs(self.bnds[i]))
                        self.x0[i] = self.bnds[i][id_closest_to_zero]

            # Remove turbines from turbs_to_opt that are downstream
            self.turbs_to_opt = (
                [i for i in self.turbs_to_opt if i not in downstream_turbines]
            )

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

    def optimize(self, verbose=True):
        """
        This method solves for the optimum turbine yaw angles for power
        production given a fixed set of atmospheric conditions
        (wind speed, direction, etc.).

        Returns:
            np.array: Optimal yaw angles for each turbine (deg).
        """
        if verbose:
            print("=====================================================")
            print("Optimizing wake redirection control...")
            print("Number of parameters to optimize = ", len(self.turbs_to_opt))
            print("=====================================================")

        opt_yaw_angles = self._optimize()

        if verbose and np.sum(opt_yaw_angles) == 0:
            print(
                "No change in controls suggested for this inflow \
                   condition..."
            )

        return opt_yaw_angles

    def reinitialize_opt(
        self,
        minimum_yaw_angle=None,
        maximum_yaw_angle=None,
        x0=None,
        bnds=None,
        opt_method=None,
        opt_options=None,
        include_unc=None,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        calc_init_power=True,
        exclude_downstream_turbines=False,
    ):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.

        Args:
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). Defaults to None.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). Defaults to None.
            x0 (iterable, optional): The initial yaw conditions (deg). If none
                are specified, they are set to the current yaw angles for all
                turbines. Defaults to None.
            bnds (iterable, optional): Bounds for the yaw angles (tuples of
                min, max values for each turbine (deg)). If none are specified,
                they are set to (minimum_yaw_angle, maximum_yaw_angle) for
                each turbine. Defaults to None.
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
            turbine_weights (iterable, optional): weighing terms that allows the
                user to emphasize power gains at particular turbines or
                completely ignore power gains from other turbines. The array
                of turbine powers from floris is multiplied with this array in
                the calculation of the objective function. Defaults to an array
                with all values 1.0 and length equal to the number of turbines.
            calc_init_power (bool, optional): If True, calculates initial wind
                farm power for each set of wind conditions. Defaults to True.
        """
        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [
                turbine.yaw_angle
                for turbine in self.fi.floris.farm.turbine_map.turbines
            ]

        self.bnds = bnds
        if bnds is not None:
            self.minimum_yaw_angle = np.min([bnds[i][0] for i in range(self.nturbs)])
            self.maximum_yaw_angle = np.max([bnds[i][1] for i in range(self.nturbs)])

            # Bound initial condition/baseline yaw angles within bounds
            self.x0 = [np.max([x, bnds[i][0]]) for i, x in enumerate(self.x0)]
            self.x0 = [np.min([x, bnds[i][1]]) for i, x in enumerate(self.x0)]
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, self.maximum_yaw_angle)

        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options
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

        if calc_init_power:
            self.fi.calculate_wake(yaw_angles=self.x0)
            turbine_powers = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options,
            )
            self.initial_farm_power = np.dot(
                self.turbine_weights,
                turbine_powers
            )

        self.exclude_downstream_turbines = exclude_downstream_turbines
        self._reduce_control_variables()


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

    @x0.setter
    def x0(self, value):
        self._x0 = value
