# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import repeat
from floris.tools.optimization.scipy.yaw_wind_rose \
	import YawOptimizationWindRose
from ....utilities import setup_logger


class YawOptimizationWindRoseParallel(YawOptimizationWindRose):
    """
    Sub class of the
    :py:class`floris.tools.optimization.YawOptimizationWindRose`
    object class that performs parallel yaw optimization for multiple wind
    speed, wind direction, turbulence intensity (optional) combinations using
    the MPIPoolExecutor method of the mpi4py.futures module.

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): 
            Interface from FLORIS to the tools package.
        minimum_yaw_angle (float, optional): Minimum constraint on 
            yaw. Defaults to None.
        maximum_yaw_angle (float, optional): Maximum constraint on 
            yaw. Defaults to None.
        minimum_ws (float, optional): Minimum wind speed at which optimization
            is performed. Assume zero power generated below this value.
            Defaults to 3 m/s.
        maximum_ws (float, optional): Maximum wind speed at which optimization
            is performed. Assume optimal yaw offsets are zero above this wind
            speed. Defaults to 25 m/s.
        wd (np.array): The wind directions for the AEP optimization.
        ws (np.array): The wind speeds for the AEP optimization.
        ti (np.array, optional): An optional list of turbulence intensity
            values for the AEP optimization. Defaults to None, meaning TI will
            not be included in the AEP calculations.
        x0 (iterable, optional): The initial yaw conditions. 
            Defaults to None. Initializes to the current turbine 
            yaw settings.
        bnds (iterable, optional): Bounds for the optimization 
            variables (pairs of min/max values for each variable). 
            Defaults to None. Initializes to [(0.0, 25.0)].
        opt_method (str, optional): The optimization method for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to 'SLSQP'.
        opt_options (dictionary, optional): Optimization options for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to {'maxiter': 100, 'disp': False,
            'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}.
        include_unc (bool): If True, uncertainty in wind direction 
            and/or yaw position is included when determining wind farm power. 
            Uncertainty is included by computing the mean wind farm power for 
            a distribution of wind direction and yaw position deviations from 
            the original wind direction and yaw angles. Defaults to False.
        unc_pmfs (dictionary, optional): A dictionary containing optional 
            probability mass functions describing the distribution of wind 
            direction and yaw position deviations when wind direction and/or 
            yaw position uncertainty is included in the power calculations. 
            Contains the following key-value pairs:  

            -   **wd_unc**: A numpy array containing wind direction deviations 
                from the original wind direction. 
            -   **wd_unc_pmf**: A numpy array containing the probability of 
                each wind direction deviation in **wd_unc** occuring. 
            -   **yaw_unc**: A numpy array containing yaw angle deviations 
                from the original yaw angles. 
            -   **yaw_unc_pmf**: A numpy array containing the probability of 
                each yaw angle deviation in **yaw_unc** occuring.

            Defaults to None, in which case default PMFs are calculated using 
            values provided in **unc_options**.
        unc_options (disctionary, optional): A dictionary containing values
            used to create normally-distributed, zero-mean probability mass
            functions describing the distribution of wind direction and yaw
            position deviations when wind direction and/or yaw position
            uncertainty is included. This argument is only used when
            **unc_pmfs** is None and contains the following key-value pairs:

            -   **std_wd**: A float containing the standard deviation of the
                    wind direction deviations from the original wind direction.
            -   **std_yaw**: A float containing the standard deviation of the
                    yaw angle deviations from the original yaw angles.
            -   **pmf_res**: A float containing the resolution in degrees of
                    the wind direction and yaw angle PMFs.
            -   **pdf_cutoff**: A float containing the cumulative distribution 
                function value at which the tails of the PMFs are truncated. 

            Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
            'pmf_res': 1.0, 'pdf_cutoff': 0.995}.

    Returns:
        YawOptimizationWindRoseParallel: An instantiated
        YawOptimizationWindRoseParallel object.
    """

    def __init__(self, fi, wd,
                           ws,
                           ti=None,
                           minimum_yaw_angle=0.0,
                           maximum_yaw_angle=25.0,
                           minimum_ws=3.0,
                           maximum_ws=25.0,
                           x0=None,
                           bnds=None,
                           opt_method='SLSQP',
                           opt_options=None,
                           include_unc=False,
                           unc_pmfs=None,
                           unc_options=None):

        self.logger = setup_logger(name=__name__)

        super().__init__(
            fi,
            wd,
            ws,
            ti=None,
            minimum_yaw_angle=0.0,
            maximum_yaw_angle=25.0,
            minimum_ws=3.0,
            maximum_ws=25.0,
            x0=None,
            bnds=None,
            opt_method='SLSQP',
            opt_options=None,
            include_unc=False,
            unc_pmfs=None,
            unc_options=None
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
                - **turbine_power_baseline** (*list* of *float* values) - A
                    list containing the baseline power without wake steering
                    for each wind turbine (W).
                - **turbine_power_no_wake** (*list* of *float* values) - A list
                    containing the ideal power without wake losses for each
                    wind turbine (W).
        """

        if ti is None:
            print('Computing wind speed = ' + str(ws) + \
                ' m/s, wind direction = ' + str(wd) + ' deg.')
        else:
            print('Computing wind speed = ' + str(ws) + \
                ' m/s, wind direction = ' + str(wd) + \
                ' deg, turbulence intensity = ' + str(ti) + '.')

        # Find baseline power in FLORIS

        if ws >= self.minimum_ws:
            if ti is None:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd,
                    wind_speed=ws
                )
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd,
                    wind_speed=ws,
                    turbulence_intensity=ti
                )
            # calculate baseline power
            self.fi.calculate_wake(yaw_angles=0.0)
            power_base = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options
            )

            # calculate power for no wake case
            self.fi.calculate_wake(no_wake=True)
            power_no_wake = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options,
                no_wake=True
            )
        else:
            power_base = self.nturbs*[0.0]
            power_no_wake = self.nturbs*[0.0]

        # add variables to dataframe
        if ti is None:
            df_base = pd.DataFrame({
                'ws':[ws],
                'wd':[wd],
                'power_baseline':[np.sum(power_base)],
                'turbine_power_baseline':[power_base],
                'power_no_wake':[np.sum(power_no_wake)],
                'turbine_power_no_wake':[power_no_wake]
            })
        else:
            df_base = pd.DataFrame({
                'ws':[ws],
                'wd':[wd],
                'ti':[ti],
                'power_baseline':[np.sum(power_base)],
                'turbine_power_baseline':[power_base],
                'power_no_wake':[np.sum(power_no_wake)],
                'turbine_power_no_wake':[power_no_wake]
            })

        return df_base

    def _optimize_one_case(self, ws, wd, ti=None):
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
                - **turbine_power_opt** (*list* of *float* values) - A list
                    containing the power produced by each wind turbine with
                    optimal yaw offsets (W).
                - **yaw_angles** (*list* of *float* values) - A list containing
                    the optimal yaw offsets for maximizing total wind farm
                    power for each wind turbine (deg).
        """

        if ti is None:
            print('Computing wind speed = ' + str(ws) + \
                ' m/s, wind direction = ' + str(wd) + ' deg.')
        else:
            print('Computing wind speed = ' + str(ws) + \
                ' m/s, wind direction = ' + str(wd) + \
                ' deg, turbulence intensity = ' + str(ti) + '.')

        # Optimizing wake redirection control

        if (ws >= self.minimum_ws) & (ws <= self.maximum_ws):
            if ti is None:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd,
                    wind_speed=ws
                )
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd,
                    wind_speed=ws,
                    turbulence_intensity=ti
                )

            opt_yaw_angles = self._optimize()

            if np.sum(opt_yaw_angles) == 0:
                print('No change in controls suggested for this inflow \
                    condition...')

            # optimized power
            self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
            power_opt = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options
            )
        elif ws >= self.minimum_ws:
            print('No change in controls suggested for this inflow \
                    condition...')
            if ti is None:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd,
                    wind_speed=ws
                )
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd,
                    wind_speed=ws,
                    turbulence_intensity=ti
                )
            self.fi.calculate_wake(yaw_angles=0.0)
            opt_yaw_angles = self.nturbs*[0.0]
            power_opt = self.fi.get_turbine_power(
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options
            )
        else:
            print('No change in controls suggested for this inflow \
                    condition...')
            opt_yaw_angles = self.nturbs*[0.0]
            power_opt = self.nturbs*[0.0]

        # add variables to dataframe
        if ti is None:
            df_opt = pd.DataFrame({
                'ws':[ws],
                'wd':[wd],
                'power_opt':[np.sum(power_opt)],
                'turbine_power_opt':[power_opt],
                'yaw_angles':[opt_yaw_angles]
            })
        else:
            df_opt = pd.DataFrame({
                'ws':[ws],
                'wd':[wd],
                'ti':[ti],
                'power_opt':[np.sum(power_opt)],
                'turbine_power_opt':[power_opt],
                'yaw_angles':[opt_yaw_angles]
            })

        return df_opt

    # Public methods    

    def calc_baseline_power(self):
        """
        For a series of (wind speed, direction, ti (optional)) combinations,
        finds the baseline power produced by the wind farm and the ideal power
        without wake losses. The optimization for different wind speed, wind
        direction combinations is parallelized using the mpi4py.futures module.

        Returns:
            - **df_base** (*Pandas DataFrame*) - DataFrame with the same number
                of rows as the length of the wd and ws arrays, containing the
                following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **ti** (*float*) - The turbulence intensity value for the
                    row. Only included if self.ti is not None.
                - **power_baseline** (*float*) - The total power produced by
                    the wind farm with baseline yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced
                    by the wind farm without wake losses (W).
                - **turbine_power_baseline** (*list* of *float* values) - A
                    list containing the baseline power without wake steering
                    for each wind turbine (W).
                - **turbine_power_no_wake** (*list* of *float* values) - A list
                    containing the ideal power without wake losses for each
                    wind turbine (W).
        """
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            err_msg = ('It appears you do not have mpi4py installed. ' + \
                'Please refer to https://mpi4py.readthedocs.io/ for ' + \
                'guidance on how to properly install the module.')
            self.logger.error(err_msg, stack_info=True)
            raise ImportError(err_msg)

        print('=====================================================')
        print('Calculating baseline power in parallel...')
        print('Number of wind conditions to calculate = ', len(self.wd))
        print('=====================================================')

        df_base = pd.DataFrame()

        with MPIPoolExecutor() as executor: 
            if self.ti is None:
                for df_base_one in executor.map(
                    self._calc_baseline_power_one_case,
                    self.ws.values,
                    self.wd.values
                ):
                
                    # add variables to dataframe
                    df_base = df_base.append(df_base_one)
            else:
                for df_base_one in executor.map(
                    self._calc_baseline_power_one_case,
                    self.ws.values,
                    self.wd.values,
                    self.ti.values
                ):
                
                    # add variables to dataframe
                    df_base = df_base.append(df_base_one)
        
        df_base.reset_index(drop=True,inplace=True)

        return df_base

    def optimize(self):
        """
        For a series of (wind speed, direction, ti (optional)) combinations,
        finds the power resulting from optimal wake steering. The optimization
        for different wind speed, wind direction combinations is parallelized
        using the mpi4py.futures module.

        Returns:
            - **df_opt** (*Pandas DataFrame*) - DataFrame with the same number
                of rows as the length of the wd and ws arrays, containing the
                following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **ti** (*float*) - The turbulence intensity value for the
                    row. Only included if self.ti is not None.
                - **power_opt** (*float*) - The total power produced by the
                    wind farm with optimal yaw offsets (W).
                - **turbine_power_opt** (*list* of *float* values) - A list
                    containing the power produced by each wind turbine with
                    optimal yaw offsets (W).
                - **yaw_angles** (*list* of *float* values) - A list containing
                    the optimal yaw offsets for maximizing total wind farm
                    power for each wind turbine (deg).
        """
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            err_msg = ('It appears you do not have mpi4py installed. ' + \
                'Please refer to https://mpi4py.readthedocs.io/ for ' + \
                'guidance on how to properly install the module.')
            self.logger.error(err_msg, stack_info=True)
            raise ImportError(err_msg)

        print('=====================================================')
        print('Optimizing wake redirection control in parallel...')
        print('Number of wind conditions to optimize = ', len(self.wd))
        print('Number of yaw angles to optimize = ', len(self.x0))
        print('=====================================================')

        df_opt = pd.DataFrame()

        with MPIPoolExecutor() as executor: 
            if self.ti is None:
                for df_opt_one in executor.map(
                    self._optimize_one_case,
                    self.ws.values,
                    self.wd.values
                ):
                
                    # add variables to dataframe
                    df_opt = df_opt.append(df_opt_one)
            else:
                for df_opt_one in executor.map(
                    self._optimize_one_case,
                    self.ws.values,
                    self.wd.values,
                    self.ti.values
                ):
            
                    # add variables to dataframe
                    df_opt = df_opt.append(df_opt_one)
        
        df_opt.reset_index(drop=True,inplace=True)

        return df_opt
