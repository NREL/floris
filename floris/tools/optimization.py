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

    def __init__(self, fi, minimum_yaw_angle=0.0,
                           maximum_yaw_angle=25.0,
                           x0=None,
                           bnds=None,
                           opt_method='SLSQP'):
        """
        Instantiate YawOptimization object and parameter values.
        """
        super().__init__()
        
        self.reinitialize_opt(
            fi=fi,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            x0=x0,
            bnds=bnds,
            opt_method=opt_method
        )

    # Private methods

    def _optimize(self, fi):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`):
                Interface from FLORIS to the tools package.

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        self.residual_plant = minimize(fi.get_power_for_yaw_angle_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                options={'eps': np.radians(5.0)})

        opt_yaw_angles = self.residual_plant.x

        return opt_yaw_angles

    def reinitialize_opt(self, fi, minimum_yaw_angle=None,
                           maximum_yaw_angle=None,
                           x0=None,
                           bnds=None,
                           opt_method=None):
        self.fi = fi
        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if opt_method is not None:
            self.opt_method = opt_method
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [turbine.yaw_angle for turbine in \
                       fi.floris.farm.turbine_map.turbines]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, 
                                 self.maximum_yaw_angle)

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in \
                     range(self.nturbs)]

    # Public methods

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
        print('=====================================================')
        print('Optimizing wake redirection control...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_yaw_angles = self._optimize(fi)

        if np.sum(opt_yaw_angles) == 0:
            print('No change in controls suggested for this inflow \
                   condition...')

        return opt_yaw_angles

    # Properties

    @property
    def minimum_yaw_angle(self):
        """
        This property gets or sets the minimum yaw angle for the 
        optimization and updates the bounds accordingly.
        
        Args:
            value (float): The minimum yaw angle (deg).

        Returns:
            minimum_yaw_angle (float): The minimum yaw angle (deg).
        """
        return self._minimum_yaw_angle

    @minimum_yaw_angle.setter
    def minimum_yaw_angle(self, value):
        if not hasattr(self, 'maximum_yaw_angle'):
            self._set_opt_bounds(value, 25.0)
        else:
            self._set_opt_bounds(value, self.maximum_yaw_angle)
        self._minimum_yaw_angle = value

    @property
    def maximum_yaw_angle(self):
        """
        This property gets or sets the maximum yaw angle for the 
        optimization and updates the bounds accordingly.
        
        Args:
            value (float): The maximum yaw angle (deg).

        Returns:
            minimum_yaw_angle (float): The maximum yaw angle (deg).
        """
        return self._maximum_yaw_angle

    @maximum_yaw_angle.setter
    def maximum_yaw_angle(self, value):
        if not hasattr(self, 'minimum_yaw_angle'):
            self._set_opt_bounds(0.0, value)
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, value)
        self._maximum_yaw_angle = value

    @property
    def x0(self):
        """
        This property gets or sets the initial yaw angles for the 
        optimization.
        
        Args:
            value (float): The initial yaw angles (deg).

        Returns:
            x0 (float): The initial yaw angles (deg).
        """
        return self._x0

    @x0.setter
    def x0(self, value):
        self._x0 = value

    @property
    def nturbs(self):
        """
        This property gets or sets the initial yaw angles for the 
        optimization.
        
        Args:
            value (float): The initial yaw angles (deg).

        Returns:
            x0 (float): The initial yaw angles (deg).
        """
        self._nturbs = len(self.fi.floris.farm.turbine_map.turbines)
        return self._nturbs

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
