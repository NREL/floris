# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from scipy.optimize import minimize

# import warnings
# warnings.simplefilter('ignore', RuntimeWarning)


class Optimization():
    """
    Base optimization class.
    """

    def __init__(self):
        self.blank = None

    def _reinitialize(self):
        pass

class YawOptimization(Optimization):
    """
    Sub class of the :py:class`floris.tools.optimization.Optimization`
    object class that performs yaw optimization.
    """

    def __init__(self):
        """
        Instantiate YawOptimization object and parameter values.
        """
        super().__init__()
        self.minimum_yaw_angle = 0.0
        self.minimum_yaw_angle = 25.0


    def optimize(self, fi, minimum_yaw_angle=None, 
                           maximum_yaw_angle=None,
                           x0=None):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`):
                Interface from FLORIS to the tools package.
            minimum_yaw_angle (float, optional): minimum constraint on yaw.
                Defaults to 0.0.
            maximum_yaw_angle (float, optional): maximum constraint on yaw.
                Defaults to 25.0.
            x0 (np.array, optional): initial yaw conditions. Defaults to 
                current turbine yaw settings.

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if x0 is None:
            self.x0 = [turbine.yaw_angle for turbine in \
                       fi.floris.farm.turbine_map.turbines]
        else:
            self.x0 = x0
        
        # initialize floris without the full flow domain; only points assigned 
        # at the turbine
        fi.floris.farm.flow_field.reinitialize_flow_field()

        # set bounds for optimization
        bnds = [(minimum_yaw_angle, maximum_yaw_angle) for turbine in \
                fi.floris.farm.turbine_map.turbines]

        print('=====================================================')
        print('Optimizing wake redirection control...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        self.residual_plant = minimize(fi.get_power_for_yaw_angle_opt,
                                self.x0,
                                method='SLSQP',
                                bounds=bnds,
                                options={'eps': np.radians(5.0)})

        if np.sum(self.residual_plant.x) == 0:
            print('No change in controls suggested for this inflow condition...')

        opt_yaw_angles = self.residual_plant.x

        return opt_yaw_angles


class LayoutOptimization(Optimization):
    """
    Sub class of the :py:class`floris.tools.optimization.Optimization`
    object class that performs layout optimization.
    """

    def __init__(self):
        """
        Instantiate LayoutOptimization object and parameter values.
        """
        super().__init__()
        self.epsilon = np.finfo(float).eps


    def optimize(self, fi):
        """
        Find optimal layout of wind turbines for power production given
        fixed atmospheric conditins (wind speed, direction, etc.).

        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`):
                Interface from FLORIS to the tools package.
        """
