# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from .optimization import Optimization
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
import pandas as pd


class YawOptimizationWindRose(Optimization):
    """
    Sub class of the :py:class`floris.tools.optimization.Optimization`
    object class that performs yaw optimization for multiple wind speed,
    wind direction, turbulence intensity (optional) combinations.

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
        wd (np.array) : The wind directions for the AEP optimization.
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
        YawOptimizationWindRose: An instantiated YawOptimizationWindRose object.
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
        """
        Instantiate YawOptimizationWindRose object and parameter values.
        """
        super().__init__(fi)
        
        if opt_options is None:
            self.opt_options = {'maxiter': 100, 'disp': False, \
                        'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}

        self.unc_pmfs = unc_pmfs

        if unc_options is None:
            self.unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                        'pmf_res': 1.0, 'pdf_cutoff': 0.995}

        self.ti = ti

        self.reinitialize_opt_wind_rose(
            wd=wd,
            ws=ws,
            ti=ti,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            minimum_ws=minimum_ws,
            maximum_ws=maximum_ws,
            x0=x0,
            bnds=bnds,
            opt_method=opt_method,
            opt_options=opt_options,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options
        )

    # Private methods

    def _get_power_for_yaw_angle_opt(self, yaw_angles):
        """
        Assign yaw angles to turbines, calculate wake, report power

        Args:
            yaw_angles (np.array): yaw to apply to each turbine

        Returns:
            power (float): wind plant power. #TODO negative? in kW?
        """

        power = -1 * self.fi.get_farm_power_for_yaw_angle(yaw_angles, \
            include_unc=self.include_unc, unc_pmfs=self.unc_pmfs, \
            unc_options=self.unc_options)

        return power / (10**3)

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        """
        Sets minimum and maximum yaw angle bounds for optimization.
        """

        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in \
                     range(self.nturbs)]

    def _optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        wind_map = self.fi.floris.farm.wind_map
        self.residual_plant = minimize(self._get_power_for_yaw_angle_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                options=self.opt_options)

        opt_yaw_angles = self.residual_plant.x
        self.fi.reinitialize_flow_field(wind_speed = wind_map.input_speed,
                              wind_direction = wind_map.input_direction,
                              turbulence_intensity = wind_map.input_ti)

        return opt_yaw_angles

    # Public methods

    def reinitialize_opt_wind_rose(self,
            wd=None,
            ws=None,
            ti=None,
            minimum_yaw_angle=None,
            maximum_yaw_angle=None,
            minimum_ws=None,
            maximum_ws=None,
            x0=None,
            bnds=None,
            opt_method=None,
            opt_options=None,
            include_unc=None,
            unc_pmfs=None,
            unc_options=None):
        """
        Reintializes parameter values for the optimization.
        
        This method reinitializes the optimization parameters and 
        bounds to the supplied values or uses what is currently stored.
        
        Args:
            fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): 
                Interface from FLORIS to the tools package.
            minimum_yaw_angle (float, optional): Minimum constraint on 
                yaw. Defaults to None.
            maximum_yaw_angle (float, optional): Maximum constraint on 
                yaw. Defaults to None.
            minimum_ws (float, optional): Minimum wind speed at which
                optimization is performed. Assume zero power generated below
                this value. Defaults to None.
            maximum_ws (float, optional): Maximum wind speed at which
                optimization is performed. Assume optimal yaw offsets are zero
                above this wind speed. Defaults to None.
            wd (np.array) : The wind directions for the AEP optimization.
            ws (np.array): The wind speeds for the AEP optimization.
            ti (np.array, optional): An optional list of turbulence intensity
                values for the AEP optimization. Defaults to None, meaning TI
                will not be included in the AEP calculations.
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
                and/or yaw position is included when determining wind farm
                power. Uncertainty is included by computing the mean wind farm
                power for a distribution of wind direction and yaw position
                deviations from the original wind direction and yaw angles.
                Defaults to None.
            unc_pmfs (dictionary, optional): A dictionary containing optional 
                probability mass functions describing the distribution of wind 
                direction and yaw position deviations when wind direction 
                and/or yaw position uncertainty is included in the power
                calculations. Contains the following key-value pairs:  

                -   **wd_unc**: A numpy array containing wind direction
                        deviations from the original wind direction. 
                -   **wd_unc_pmf**: A numpy array containing the probability of 
                        each wind direction deviation in **wd_unc** occuring. 
                -   **yaw_unc**: A numpy array containing yaw angle deviations 
                        from the original yaw angles. 
                -   **yaw_unc_pmf**: A numpy array containing the probability
                        of each yaw angle deviation in **yaw_unc** occuring.

                Defaults to None. If the object's **include_unc** parameter is
                True and **unc_pmfs** has not been initialized, initializes
                using normally-distributed, zero-mean PMFs based on the values
                in **unc_options**.
            unc_options (disctionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of
                        the wind direction deviations from the original wind
                        direction.
                -   **std_yaw**: A float containing the standard deviation of
                        the yaw angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees
                        of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative
                        distribution function value at which the tails of the
                        PMFs are truncated. 

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw':
                1.75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
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
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [turbine.yaw_angle for turbine in \
                       self.fi.floris.farm.turbine_map.turbines]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, 
                                 self.maximum_yaw_angle)
        if include_unc is not None:
            self.include_unc = include_unc
        if unc_pmfs is not None:
            self.unc_pmfs = unc_pmfs
        if unc_options is not None:
            self.unc_options = unc_options

        if self.include_unc & (self.unc_pmfs is None):
            if self.unc_options is None:
                self.unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                            'pmf_res': 1.0, 'pdf_cutoff': 0.995}

            # create normally distributed wd and yaw uncertaitny pmfs
            if self.unc_options['std_wd'] > 0:
                wd_bnd = int(
                    np.ceil(
                        norm.ppf(
                            self.unc_options['pdf_cutoff'],
                            scale= self.unc_options['std_wd']
                        )/self.unc_options['pmf_res']
                    )
                )
                wd_unc = np.linspace(
                    -1*wd_bnd*self.unc_options['pmf_res'],
                    wd_bnd*self.unc_options['pmf_res'],
                    2*wd_bnd+1
                )
                wd_unc_pmf = norm.pdf(wd_unc,scale=self.unc_options['std_wd'])
                # normalize so sum = 1.0
                wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf) 
            else:
                wd_unc = np.zeros(1)
                wd_unc_pmf = np.ones(1)

            if self.unc_options['std_yaw'] > 0:
                yaw_bnd = int(
                    np.ceil(
                        norm.ppf(
                            self.unc_options['pdf_cutoff'],
                            scale=self.unc_options['std_yaw']
                        )/self.unc_options['pmf_res']
                    )
                )
                yaw_unc = np.linspace(
                    -1*yaw_bnd*self.unc_options['pmf_res'],
                    yaw_bnd*self.unc_options['pmf_res'],
                    2*yaw_bnd+1
                )
                yaw_unc_pmf = norm.pdf(
                    yaw_unc,scale=self.unc_options['std_yaw']
                )
                # normalize so sum = 1.0
                yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf) 
            else:
                yaw_unc = np.zeros(1)
                yaw_unc_pmf = np.ones(1)

            self.unc_pmfs = {'wd_unc': wd_unc, 'wd_unc_pmf': wd_unc_pmf, \
                        'yaw_unc': yaw_unc, 'yaw_unc_pmf': yaw_unc_pmf}

    def calc_baseline_power(self):
        """
        For a series of (wind speed, direction, ti (optional)) combinations,
        finds the baseline power produced by the wind farm and the ideal power
        without wake losses. 

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
        print('=====================================================')
        print('Calculating baseline power...')
        print('Number of wind conditions to calculate = ', len(self.wd))
        print('=====================================================')

        # Put results in dict for speed, instead of previously
        # appending to frame.
        result_dict = dict()

        for i in range(len(self.wd)):
            if self.ti is None:
                print('Computing wind speed, wind direction pair ' + str(i) \
                    + ' out of ' + str(len(self.wd)) + ': wind speed = ' \
                    + str(self.ws[i]) + ' m/s, wind direction = ' \
                    + str(self.wd[i])+' deg.')
            else:
                print('Computing wind speed, wind direction, turbulence ' \
                    + 'intensity set ' + str(i) + ' out of ' \
                    + str(len(self.wd)) + ': wind speed = ' + str(self.ws[i]) \
                    + ' m/s, wind direction = ' + str(self.wd[i]) \
                    + ' deg, turbulence intensity = '+str(self.ti[i])+'.')

            # Find baseline power in FLORIS

            if self.ws[i] >= self.minimum_ws:
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]])
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i]
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
            if self.ti is None:
                result_dict[i] = {
                    'ws':self.ws[i],
                    'wd':self.wd[i],
                    'power_baseline':np.sum(power_base),
                    'turbine_power_baseline':power_base,
                    'power_no_wake':np.sum(power_no_wake),
                    'turbine_power_no_wake':power_no_wake
                }
                # df_base = df_base.append(pd.DataFrame(
                #    {'ws':[self.ws[i]],'wd':[self.wd[i]],
                #    'power_baseline':[np.sum(power_base)],
                #    'turbine_power_baseline':[power_base],
                #    'power_no_wake':[np.sum(power_no_wake)],
                #    'turbine_power_no_wake':[power_no_wake]}))
            else:
                result_dict[i] = {
                    'ws':self.ws[i],
                    'wd':self.wd[i],
                    'ti':self.ti[i],
                    'power_baseline':np.sum(power_base),
                    'turbine_power_baseline':power_base,
                    'power_no_wake':np.sum(power_no_wake),
                    'turbine_power_no_wake':power_no_wake
                }
                # df_base = df_base.append(pd.DataFrame(
                #    {'ws':[self.ws[i]],'wd':[self.wd[i]],
                #    'ti':[self.ti[i]],'power_baseline':[np.sum(power_base)],
                #    'turbine_power_baseline':[power_base],
                #    'power_no_wake':[np.sum(power_no_wake)],
                #    'turbine_power_no_wake':[power_no_wake]}))
        df_base = pd.DataFrame.from_dict(result_dict, "index")
        df_base.reset_index(drop=True,inplace=True)

        return df_base

    def optimize(self):
        """
        For a series of (wind speed, direction, ti (optional)) combinations,
        finds the power resulting from optimal wake steering. 

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
        print('=====================================================')
        print('Optimizing wake redirection control...')
        print('Number of wind conditions to optimize = ', len(self.wd))
        print('Number of yaw angles to optimize = ', len(self.x0))
        print('=====================================================')

        df_opt = pd.DataFrame()

        for i in range(len(self.wd)):
            if self.ti is None:
                print('Computing wind speed, wind direction pair ' + str(i) \
                    + ' out of ' + str(len(self.wd)) + ': wind speed = ' \
                    + str(self.ws[i]) + ' m/s, wind direction = ' \
                    + str(self.wd[i])+' deg.')
            else:
                print('Computing wind speed, wind direction, turbulence ' \
                    + 'intensity set ' + str(i) + ' out of ' \
                    + str(len(self.wd)) + ': wind speed = ' + str(self.ws[i]) \
                    + ' m/s, wind direction = ' + str(self.wd[i]) \
                    + ' deg, turbulence intensity = ' + str(self.ti[i]) + '.')

            # Optimizing wake redirection control

            if (self.ws[i] >= self.minimum_ws) \
                                            & (self.ws[i] <= self.maximum_ws):
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i]
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
            elif self.ws[i] >= self.maximum_ws:
                print('No change in controls suggested for this inflow \
                        condition...')
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]])
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i]
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
            if self.ti is None:
                df_opt = df_opt.append(pd.DataFrame({
                    'ws':[self.ws[i]],
                    'wd':[self.wd[i]],
                    'power_opt':[np.sum(power_opt)],
                    'turbine_power_opt':[power_opt],
                    'yaw_angles':[opt_yaw_angles]
                }))
            else:
                df_opt = df_opt.append(pd.DataFrame({
                    'ws':[self.ws[i]],
                    'wd':[self.wd[i]],
                    'ti':[self.ti[i]],
                    'power_opt':[np.sum(power_opt)],
                    'turbine_power_opt':[power_opt],
                    'yaw_angles':[opt_yaw_angles]
                }))

        df_opt.reset_index(drop=True,inplace=True)

        return df_opt
