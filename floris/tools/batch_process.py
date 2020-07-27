# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# In theory, this could be a member function of FLORIS interface
# But for now seperating in case we want additional libraries here

import numpy as np


def batch_simulate(fi, ws_array, wd_array, ti_array=None):
    """
    Given arrays of floris inputs, and a floris interface
    Compute a list of FLORIS powers
    Args:
        fi (FLORIS): Floris interface to simulate with
        ws_array (np.array): Values of wind speed
        wd_array (np.array): Values of wind direction
        ti_array (np.array, optional): Values of wind direction
    Returns:
        np.array: num_sim x num_turbine array of power outputs
    """

    num_sims = len(ws_array)
    num_turbines = len(fi.floris.farm.turbines)
    turbine_powers = np.zeros([num_sims, num_turbines])

    # If ti_array is none, then assume fixed value
    if ti_array is None:
        ti_array = np.ones_like(ws_array) * fi.floris.farm.wind_map.input_ti

    # Collect the results
    for idx, (ws, wd, ti) in enumerate(zip(ws_array, wd_array, ti_array)):

        # Collect the results
        fi.reinitialize_flow_field(
            wind_speed=ws, wind_direction=wd, turbulence_intensity=ti
        )
        fi.calculate_wake()
        turbine_powers[idx, :] = np.array(fi.get_turbine_power())

    return turbine_powers
