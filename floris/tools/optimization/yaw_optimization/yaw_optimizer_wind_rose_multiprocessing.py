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


import multiprocessing as mp

import pandas as pd

from .yaw_optimizer_wind_rose_serial import YawOptimizationWindRose


class YawOptimizationWindRoseParallel(YawOptimizationWindRose):
    """
    YawOptimizationWindRoseParallel is a subclass of
    :py:class:`~.tools.optimization.scipy.general_library
    .yaw_wind_rose_serial_wrapper.YawOptimizationWindRose` that
    is used to optimize the yaw angles of all turbines in a Floris Farm for
    multiple sets of inflow conditions (combinations of wind speed, wind direction,
    and optionally turbulence intensity) using the provid yaw_optimization_obj
    object. Calculations are performed in parallel using the multiprocessing module.
    """

    def __init__(
        self,
        yaw_optimization_obj,
        wd_array,
        ws_array,
        ti_array=None,
        minimum_ws=0.0,
        maximum_ws=25.0,
        num_processes=None,
        verbose=True,
    ):
        """
        Instantiate YawOptimizationWindRoseParallel object with a yaw optimization
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
            num_processes (int, optional): Number of processes to use in the
                multiprocessing pool. If None specified, automatically determines the
                number of cores available on your system. Defaults to None.
            verbose (bool, optional): If True, print progress and information about
                the optimization. Useful for debugging. Defaults to True.
        """

        super().__init__(
            yaw_optimization_obj=yaw_optimization_obj,
            wd_array=wd_array,
            ws_array=ws_array,
            ti_array=ti_array,
            minimum_ws=minimum_ws,
            maximum_ws=maximum_ws,
            verbose=verbose
        )

        if num_processes is None:
            num_processes = mp.cpu_count()
            print("Using %d cores for parallel processing." % num_processes)

        self.num_processes = num_processes

    # Public methods

    def optimize(self):
        """
        This method solves for the optimum turbine yaw angles for power
        production and the resulting power produced by the wind farm for a
        series of wind speed, wind direction, and optionally TI combinations.
        Calculations are parallelized using the multiprocessing module.

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
                - **power_baseline_weighted** (*float*) - The total power produced
                by the wind farm with the baseline yaw offsets weighted by the
                turbine weights specified by the user (W).
                - **turbine_power_baseline** (*float*) - The power produced
                by each turbine in the wind farm with the baseline yaw offsets (W).
                - **power_opt** (*float*) - The total power produced by the
                wind farm with optimal yaw offsets (W).
                - **power_opt_weighted** (*float*) - The total power produced
                by the wind farm with the optimal yaw offsets weighted by the
                turbine weights specified by the user (W).
                - **turbine_power_opt** (*float*) - The power produced by each
                turbine in the wind farm with the optimal yaw offsets (W).
                - **yaw_angles** (*list* (*float*)) - A list containing
                the optimal yaw offsets for maximizing total wind farm power
                for each wind turbine (deg).
        """

        print("=====================================================")
        print(
            "Optimizing wake redirection control in parallel using multiprocessing..."
        )
        print("Number of wind conditions to optimize = ", len(self.wd_array))
        print("=====================================================")

        # Enforce calculation of baseline conditions
        self.yaw_opt.calc_init_power = True

        # Set up arguments
        args = []
        for i in range(len(self.wd_array)):
            wd = self.wd_array[i]
            ws = self.ws_array[i]
            ti = self.ti_array[i]
            args.append([wd, ws, ti])

        with mp.Pool(self.num_processes) as p:
            df_list = p.starmap(self._optimize_one_case, args)

        df_opt = pd.concat(df_list, ignore_index=True)
        df_opt = df_opt.reset_index(drop=True)
        return df_opt
