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

from itertools import repeat

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ....logging_manager import LoggerBase
from .yaw_wind_rose import YawOptimizationWindRose


class YawOptimizationWindRoseParallel(YawOptimizationWindRose, LoggerBase):
    """
    YawOptimizationWindRose is a subclass of
    :py:class:`~.tools.optimizationscipy.YawOptimizationWindRose` that is used
    to perform parallel computing to optimize the yaw angles of all turbines in
    a Floris Farm for multiple sets of inflow conditions (combinations of wind
    speed, wind direction, and optionally turbulence intensity) using the scipy
    optimize package. Parallel optimization is performed using the
    MPIPoolExecutor method of the mpi4py.futures module.
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
        exclude_downstream_turbines=False,
    ):
        """
        Instantiate YawOptimizationWindRoseParallel object with a
        FlorisInterface object and assign parameter values.

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
            exclude_downstream_turbines (bool, optional): If True,
                automatically finds and excludes turbines that are most
                downstream from the optimization problem. This significantly
                reduces computation time at no loss in performance. The yaw
                angles of these downstream turbines are fixed to 0.0 deg if
                the yaw bounds specified in self.bnds allow that, or otherwise
                are fixed to the lower or upper yaw bound, whichever is closer
                to 0.0. Defaults to False.
        """
        super().__init__(
            fi,
            wd,
            ws,
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
            calc_init_power=False,
            exclude_downstream_turbines=exclude_downstream_turbines,
        )

    # Private methods

    def _calc_baseline_power_one_case(self, ws, wd, ti=None):
        """
        For a single (wind speed, direction, ti (optional)) combination, finds
        the baseline power produced by the wind farm and the ideal power
        without wake losses.

        Args:
            ws (float): The wind speed used in floris for the yaw optimization.
            wd (float): The wind direction used in floris for the yaw
                optimization.
            ti (float, optional): An optional turbulence intensity value for
                the yaw optimization. Defaults to None, meaning TI will not be
                included in the AEP calculations.

        Returns:
            - **df_base** (*Pandas DataFrame*) - DataFrame with a single row,
                containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **ti** (*float*) - The turbulence intensity value for the
                    row. Only included if self.ti is not None.
                - **power_baseline** (*float*) - The total power produced by
                    the wind farm with baseline yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced
                    by the wind farm without wake losses (W).
                - **turbine_power_baseline** (*list* (*float*)) - A
                    list containing the baseline power without wake steering
                    for each wind turbine (W).
                - **turbine_power_no_wake** (*list* (*float*)) - A list
                    containing the ideal power without wake losses for each
                    wind turbine (W).
        """
        if ti is None:
            print(
                "Computing wind speed = "
                + str(ws)
                + " m/s, wind direction = "
                + str(wd)
                + " deg."
            )
        else:
            print(
                "Computing wind speed = "
                + str(ws)
                + " m/s, wind direction = "
                + str(wd)
                + " deg, turbulence intensity = "
                + str(ti)
                + "."
            )

        # Find baseline power in FLORIS

        if ws >= self.minimum_ws:
            if ti is None:
                self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd, wind_speed=ws, turbulence_intensity=ti
                )
            # calculate baseline power
            self.fi.calculate_wake(yaw_angles=self.yaw_angles_baseline)
            power_base = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options,
            )
            # calculate power for no wake case
            self.fi.calculate_wake(yaw_angles=self.yaw_angles_baseline, no_wake=True)
            power_no_wake = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options,
                no_wake=True,
            )
        else:
            power_base = self.nturbs * [0.0]
            power_no_wake = self.nturbs * [0.0]

        # Add turbine weighing terms
        power_base = np.multiply(self.turbine_weights, power_base)
        power_no_wake = np.multiply(self.turbine_weights, power_no_wake)

        # add variables to dataframe
        if ti is None:
            df_base = pd.DataFrame(
                {
                    "ws": [ws],
                    "wd": [wd],
                    "power_baseline": [np.sum(power_base)],
                    "turbine_power_baseline": [power_base],
                    "power_no_wake": [np.sum(power_no_wake)],
                    "turbine_power_no_wake": [power_no_wake],
                }
            )
        else:
            df_base = pd.DataFrame(
                {
                    "ws": [ws],
                    "wd": [wd],
                    "ti": [ti],
                    "power_baseline": [np.sum(power_base)],
                    "turbine_power_baseline": [power_base],
                    "power_no_wake": [np.sum(power_no_wake)],
                    "turbine_power_no_wake": [power_no_wake],
                }
            )

        return df_base

    def _optimize_one_case(self, ws, wd, initial_farm_power, ti=None):
        """
        For a single (wind speed, direction, ti (optional)) combination, finds
        the power resulting from optimal wake steering.

        Args:
            ws (float): The wind speed used in floris for the yaw optimization.
            wd (float): The wind direction used in floris for the yaw
                optimization.
            ti (float, optional): An optional turbulence intensity value for
                the yaw optimization. Defaults to None, meaning TI will not be
                included in the AEP calculations.

        Returns:
            - **df_opt** (*Pandas DataFrame*) - DataFrame with a single row,
                containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **ti** (*float*) - The turbulence intensity value for the
                    row. Only included if self.ti is not None.
                - **power_opt** (*float*) - The total power produced by the
                    wind farm with optimal yaw offsets (W).
                - **turbine_power_opt** (*list* (*float*)) - A list
                    containing the power produced by each wind turbine with
                    optimal yaw offsets (W).
                - **yaw_angles** (*list* (*float*)) - A list containing
                    the optimal yaw offsets for maximizing total wind farm
                    power for each wind turbine (deg).
        """
        if ti is None:
            print(
                "Computing wind speed = "
                + str(ws)
                + " m/s, wind direction = "
                + str(wd)
                + " deg."
            )
        else:
            print(
                "Computing wind speed = "
                + str(ws)
                + " m/s, wind direction = "
                + str(wd)
                + " deg, turbulence intensity = "
                + str(ti)
                + "."
            )

        # Optimizing wake redirection control

        if (ws >= self.minimum_ws) & (ws <= self.maximum_ws):
            if ti is None:
                self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd, wind_speed=ws, turbulence_intensity=ti
                )

            self.initial_farm_power = initial_farm_power
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
        elif ws >= self.minimum_ws:
            print(
                "No change in controls suggested for this inflow \
                    condition..."
            )
            if ti is None:
                self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd, wind_speed=ws, turbulence_intensity=ti
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

        # Add turbine weighing terms
        power_opt = np.multiply(self.turbine_weights, power_opt)

        # add variables to dataframe
        if ti is None:
            df_opt = pd.DataFrame(
                {
                    "ws": [ws],
                    "wd": [wd],
                    "power_opt": [np.sum(power_opt)],
                    "turbine_power_opt": [power_opt],
                    "yaw_angles": [opt_yaw_angles],
                }
            )
        else:
            df_opt = pd.DataFrame(
                {
                    "ws": [ws],
                    "wd": [wd],
                    "ti": [ti],
                    "power_opt": [np.sum(power_opt)],
                    "turbine_power_opt": [power_opt],
                    "yaw_angles": [opt_yaw_angles],
                }
            )

        return df_opt

    # Public methods

    def calc_baseline_power(self):
        """
        This method computes the baseline power produced by the wind farm and
        the ideal power without wake losses for a series of wind speed, wind
        direction, and optionally TI combinations. The optimization for
        different wind condition combinations is parallelized using the mpi4py
        futures module.

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
                he wind farm with baseline yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced
                by the wind farm without wake losses (W).
                - **turbine_power_baseline** (*list* (*float*)) - A list
                containing the baseline power without wake steering for each
                wind turbine in the wind farm (W).
                - **turbine_power_no_wake** (*list* (*float*)) - A list
                containing the ideal power without wake losses for each wind
                turbine in the wind farm (W).
        """
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            err_msg = (
                "It appears you do not have mpi4py installed. "
                + "Please refer to https://mpi4py.readthedocs.io/ for "
                + "guidance on how to properly install the module."
            )
            self.logger.error(err_msg, stack_info=True)
            raise ImportError(err_msg)

        print("=====================================================")
        print("Calculating baseline power in parallel...")
        print("Number of wind conditions to calculate = ", len(self.wd))
        print("=====================================================")

        df_base = pd.DataFrame()

        with MPIPoolExecutor() as executor:
            if self.ti is None:
                for df_base_one in executor.map(
                    self._calc_baseline_power_one_case, self.ws.values, self.wd.values
                ):

                    # add variables to dataframe
                    df_base = df_base.append(df_base_one)
            else:
                for df_base_one in executor.map(
                    self._calc_baseline_power_one_case,
                    self.ws.values,
                    self.wd.values,
                    self.ti.values,
                ):

                    # add variables to dataframe
                    df_base = df_base.append(df_base_one)

        df_base.reset_index(drop=True, inplace=True)

        self.df_base = df_base
        return df_base

    def optimize(self):
        """
        This method solves for the optimum turbine yaw angles for power
        production and the resulting power produced by the wind farm for a
        series of wind speed, wind direction, and optionally TI combinations.
        The optimization for different wind condition combinations is
        parallelized using the mpi4py.futures module.

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
                - **turbine_power_opt** (*list* (*float*)) - A list containing
                the power produced by each wind turbine with optimal yaw
                offsets (W).
                - **yaw_angles** (*list* (*float*)) - A list containing the
                optimal yaw offsets for maximizing total wind farm power for
                each wind turbine (deg).
        """
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            err_msg = (
                "It appears you do not have mpi4py installed. "
                + "Please refer to https://mpi4py.readthedocs.io/ for "
                + "guidance on how to properly install the module."
            )
            self.logger.error(err_msg, stack_info=True)
            raise ImportError(err_msg)

        print("=====================================================")
        print("Optimizing wake redirection control in parallel...")
        print("Number of wind conditions to optimize = ", len(self.wd))
        print("Number of yaw angles to optimize = ", len(self.turbs_to_opt))
        print("=====================================================")

        df_opt = pd.DataFrame()

        with MPIPoolExecutor() as executor:
            if self.ti is None:
                for df_opt_one in executor.map(
                    self._optimize_one_case,
                    self.ws.values,
                    self.wd.values,
                    self.df_base.power_baseline.values,
                ):

                    # add variables to dataframe
                    df_opt = df_opt.append(df_opt_one)
            else:
                for df_opt_one in executor.map(
                    self._optimize_one_case,
                    self.ws.values,
                    self.wd.values,
                    self.df_base.power_baseline.values,
                    self.ti.values,
                ):

                    # add variables to dataframe
                    df_opt = df_opt.append(df_opt_one)

        df_opt.reset_index(drop=True, inplace=True)

        return df_opt
