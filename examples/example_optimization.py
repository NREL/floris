"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from floris.floris import Floris
import floris.optimization as flopt
import numpy as np

# setup floris and process input file
floris = Floris("example_input.json")

# run FLORIS with no yaw
turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
for k,turbine in enumerate(turbines):
    turbine.yaw_angle = 0.0
floris.farm.flow_field.calculate_wake()
power_initial = np.sum([turbine.power for turbine in turbines])  # determine initial power production

# number of turbines
nTurbs = len(turbines)

# set bounds for the optimization on the yaw angles (deg)
minimum_yaw_angle = 0.0
maximum_yaw_angle = 25.0

# compute the optimal yaw angles
opt_yaw_angles = flopt.wake_steering(floris, minimum_yaw_angle,
                                     maximum_yaw_angle)

print('Optimal yaw angles for:')
for i, yaw in enumerate(opt_yaw_angles):
    print('Turbine ', i, ' yaw angle = ', np.degrees(yaw))

# calculate power gain
# assign yaw angles to turbines
turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
for i,turbine in enumerate(turbines):
    turbine.yaw_angle = opt_yaw_angles[i]

# compute the new wake with yaw angles
floris.farm.flow_field.calculate_wake()

# optimal power
power_opt = np.sum([turbine.power for turbine in turbines])

print('Power increased by ', 100*(power_opt-power_initial)/power_initial, '%')
