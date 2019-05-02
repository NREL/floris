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


def optimize_yaw(fi, minimum_yaw_angle=0.0, maximum_yaw_angle=25.0):
    """
    Find optimum setting of turbine yaw angles for power production
    given fixed atmospheric conditins (wind speed, direction, etc.)

    Args:
        fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`):
            Interface from FLORIS to the wfc tools
        minimum_yaw_angle (float, optional): minimum constraint on yaw.
            Defaults to 0.0.
        maximum_yaw_angle (float, optional): maximum constraint on yaw.
            Defaults to 25.0.

    Returns:
        opt_yaw_angles (np.array): optimal yaw angles of each turbine.
    """
    # initialize floris without the full flow domain; only points assigned at the turbine
    fi.floris.farm.flow_field.reinitialize_flow_field()

    # set initial conditions
    x0 = []
    bnds = []

    turbines = fi.floris.farm.turbine_map.turbines
    x0 = [turbine.yaw_angle for turbine in turbines]
    bnds = [(minimum_yaw_angle, maximum_yaw_angle) for turbine in turbines]

    print('=====================================================')
    print('Optimizing wake redirection control...')
    print('Number of parameters to optimize = ', len(x0))
    print('=====================================================')

    residual_plant = minimize(fi.get_power_for_yaw_angle_opt,
                              x0,
                              method='SLSQP',
                              bounds=bnds,
                              options={'eps': np.radians(5.0)})

    if np.sum(residual_plant.x) == 0:
        print('No change in controls suggested for this inflow condition...')

    # %%
    opt_yaw_angles = residual_plant.x

    return opt_yaw_angles
