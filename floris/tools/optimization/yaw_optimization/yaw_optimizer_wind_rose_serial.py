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


class YawOptimizationWindRose:
    """
    YawOptimizationWindRose is a subclass of
    :py:class:`~.tools.optimization.scipy.general_library.YawOptimization` that
    is used to optimize the yaw angles of all turbines in a Floris Farm for
    multiple sets of inflow conditions (combinations of wind speed, wind direction,
    and optionally turbulence intensity) using the provid yaw_optimization_obj
    object. Calculations are performed in serial. For parallelization, see
    `.yaw_wind_rose_wrapper.YawOptimizationWindRoseParallel`.
    """

    def __init__(
        self,
        yaw_optimization_obj,
        wd_array,
        ws_array,
        ti_array=None,
        minimum_ws=0.0,
        maximum_ws=25.0,
        verbose=True,
    ):
        """
        Instantiate YawOptimizationWindRose object with a yaw optimization
        object and assign parameter values.

        Args:
            yaw_optimization_obj (class built upon :py:class:`~.tools.optimization.
            general_library.YawOptimization`, for example `YawOptimizationScipy`):
                This object is used to optimize the yaw angles for each subset of
                ambient conditions.
            wd_array (iterable) : The wind directions for which the yaw angles are
                optimized (deg).
            ws_array (iterable): The wind speeds for which the yaw angles are
                optimized (m/s).
            ti_array (iterable, optional): An optional list of turbulence intensity
                values for which the yaw angles are optimized. If not
                specified, the current TI value in the Floris object will be
                used for all optimizations. Defaults to None.
            minimum_ws (float, optional): Lower bound on the wind speed for which
                yaw angles are to be optimized. If the ambient wind speed is below
                this value, the optimal yaw angles will default to the baseline
                yaw angles. If None is specified, defaults to 0.0 (m/s).
            maximum_ws (float, optional): Upper bound on the wind speed for which
                yaw angles are to be optimized. If the ambient wind speed is above
                this value, the optimal yaw angles will default to the baseline
                yaw angles. If None is specified, defaults to 25.0 (m/s).
            verbose (bool, optional): If True, print progress and information about
                the optimization. Useful for debugging. Defaults to True.
        """

        self.yaw_opt = yaw_optimization_obj
        self.wd_array = wd_array
        self.ws_array = ws_array
        if ti_array is None:
            ti_ambient = np.min(
                self.yaw_opt.fi.floris.farm.turbulence_intensity
            )
            self.ti_array = np.ones_like(wd_array) * ti_ambient
        else:
            self.ti_array = ti_array

        self.minimum_ws = minimum_ws
        self.maximum_ws = maximum_ws

        self.verbose = verbose

    # Public methods

    def plot_clusters(self):
        # Save initial wind direction and loop through array
        wds_init = self.yaw_opt.fi.floris.farm.wind_direction
        for wd in self.wd_array:
            self.yaw_opt.reinitialize_flow_field(wind_direction=wd)
            self.yaw_opt.plot_clusters()

        # Restore initial wind directions
        self.yaw_opt.reinitialize_flow_field(wind_direction=wds_init)

    def _optimize_one_case(self, wd, ws, ti):
        if self.verbose:
            print(
                "  Computing optimal yaw angles for"
                + " wind speed = %.2f m/s, " % ws
                + " wind direction = %.2f deg" % wd
                + " turbulence intensity = %.3f." % ti
            )

        # Optimizing wake redirection control
        self.yaw_opt.reinitialize_flow_field(
            wind_direction=[wd], wind_speed=[ws], turbulence_intensity=[ti]
        )
        if (ws >= self.minimum_ws) and (ws <= self.maximum_ws):
            opt_yaw_angles = self.yaw_opt.optimize()
        else:
            print("   Skipping optimization: outside of wind speed bounds.")
            opt_yaw_angles = self.yaw_opt.yaw_angles_baseline

        # Calculate baseline power
        self.yaw_opt.fi.calculate_wake(
            yaw_angles=self.yaw_opt.yaw_angles_baseline
        )
        power_turbs_base = self.yaw_opt.fi.get_turbine_power(
            include_unc=self.yaw_opt.include_unc,
            unc_pmfs=self.yaw_opt.unc_pmfs,
            unc_options=self.yaw_opt.unc_options,
        )

        # Calculate baseline power without wake losses
        self.yaw_opt.fi.calculate_wake(
            yaw_angles=self.yaw_opt.yaw_angles_baseline, no_wake=True
        )
        power_turbs_base_nowakes = self.yaw_opt.fi.get_turbine_power(
            include_unc=self.yaw_opt.include_unc,
            unc_pmfs=self.yaw_opt.unc_pmfs,
            unc_options=self.yaw_opt.unc_options,
            no_wake=True,
        )

        # Calculate optimized power
        self.yaw_opt.fi.calculate_wake(yaw_angles=opt_yaw_angles)
        power_turbs_opt = self.yaw_opt.fi.get_turbine_power(
            include_unc=self.yaw_opt.include_unc,
            unc_pmfs=self.yaw_opt.unc_pmfs,
            unc_options=self.yaw_opt.unc_options,
        )

        # Calculate optimized power without wake losses
        self.yaw_opt.fi.calculate_wake(
            yaw_angles=opt_yaw_angles, no_wake=True
        )
        power_turbs_opt_nowakes = self.yaw_opt.fi.get_turbine_power(
            include_unc=self.yaw_opt.include_unc,
            unc_pmfs=self.yaw_opt.unc_pmfs,
            unc_options=self.yaw_opt.unc_options,
            no_wake=True,
        )

        # Return a dataframe with outputs
        w = self.yaw_opt.turbine_weights
        return pd.DataFrame(
            {
                "ws": [ws],
                "wd": [wd],
                "ti": [ti],
                "power_baseline": [np.sum(power_turbs_base)],
                "power_baseline_nowakes": [np.sum(power_turbs_base_nowakes)],
                "power_baseline_weighted": [np.dot(w, power_turbs_base)],
                "power_baseline_weighted_nowakes": [
                    np.dot(w, power_turbs_base_nowakes)
                ],
                "turbine_power_baseline": [power_turbs_base],
                "turbine_power_baseline_nowakes": [power_turbs_base_nowakes],
                "power_opt": [np.sum(power_turbs_opt)],
                "power_opt_nowakes": [np.sum(power_turbs_opt_nowakes)],
                "power_opt_weighted": [np.dot(w, power_turbs_opt)],
                "power_opt_weighted_nowakes": [
                    np.dot(w, power_turbs_opt_nowakes)
                ],
                "turbine_power_opt": [power_turbs_opt],
                "turbine_power_opt_nowakes": [power_turbs_opt_nowakes],
                "yaw_angles": [opt_yaw_angles],
            }
        )

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
                included if self.ti_array is not None.
                - **power_baseline** (*float*) - The total power produced by the
                wind farm with the baseline yaw offsets (W).
                - **power_baseline_nowakes** (*float*) - The total power produced
                by the wind farm with the baseline yaw offsets when the wake losses
                are assumed to be zero (W).
                - **power_baseline_weighted** (*float*) - The total power produced
                by the wind farm with the baseline yaw offsets weighted by the
                turbine weights specified by the user (W).
                - **power_baseline_weighted_nowakes** (*float*) - The total power
                produced by the wind farm with the baseline yaw offsets when the
                wake losses are assumed to be zero, and weighted by the turbine
                weights specified by the user (W).
                - **turbine_power_baseline** (*float*) - The power produced
                by each turbine in the wind farm with the baseline yaw offsets (W).
                - **turbine_power_baseline_nowakes** (*float*) - The power produced
                by each turbine in the wind farm with the baseline yaw offsets, and
                when the wake losses are assumed to be zero (W).
                - **power_opt** (*float*) - The total power produced by the wind
                farm with optimal yaw offsets (W).
                - **power_opt_nowakes** (*float*) - The total power produced by the
                wind farm with optimal yaw offsets when the wake losses are assumed
                to be zero (W).
                - **power_opt_weighted** (*float*) - The total power produced by the
                wind farm with the optimal yaw offsets weighted by the turbine
                weights specified by the user (W).
                - **power_opt_weighted_nowakes** (*float*) - The total power
                produced by the wind farm with the optimal yaw offsets when the
                wake losses are assumed to be zero, and weighted by the turbine
                weights specified by the user (W).
                - **turbine_power_opt** (*float*) - The power produced by each
                turbine in the wind farm with the optimal yaw offsets (W).
                - **turbine_power_opt_nowakes** (*float*) - The power produced by
                each turbine in the wind farm when the wake losses are assumed to
                be zero, and with the optimal yaw offsets (W).
                - **yaw_angles** (*list* (*float*)) - A list containing
                the optimal yaw offsets for maximizing total wind farm power
                for each wind turbine (deg).
        """
        print("=====================================================")
        print("Optimizing wake redirection control by serial processing...")
        print("Number of wind conditions to optimize = ", len(self.wd_array))
        print("=====================================================")

        # Enforce calculation of baseline conditions
        self.yaw_opt.calc_init_power = True

        # Initialize empty dataframe
        df_opt = pd.DataFrame()

        for i in range(len(self.wd_array)):
            wd = self.wd_array[i]
            ws = self.ws_array[i]
            ti = self.ti_array[i]

            # Optimize case and append to df_opt dataframe
            df_i = self._optimize_one_case(wd=wd, ws=ws, ti=ti)
            df_opt = df_opt.append(df_i)

        df_opt = df_opt.reset_index(drop=True)
        return df_opt
