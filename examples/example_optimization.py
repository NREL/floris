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

from floris import Floris
import floris.optimization as flopt
import numpy as np

# Setup floris with the Gauss velocity and deflection models
floris = Floris("example_input.json")
floris.farm.set_wake_model("gauss")

# Run floris with no yaw
floris.farm.set_yaw_angles(0.0, calculate_wake=True)

# Determine initial power production
power_initial = np.sum([turbine.power for turbine in floris.farm.turbines])

# Set bounds for the optimization on the yaw angles (deg)
minimum_yaw, maximum_yaw = 0.0, 25.0

# Compute the optimal yaw angles
opt_yaw_angles = flopt.wake_steering(floris, minimum_yaw, maximum_yaw)
print('Optimal yaw angles for:')
for i, yaw in enumerate(opt_yaw_angles):
    print('Turbine ', i, ' yaw angle = ', np.degrees(yaw))

# Calculate power gain with new yaw angles
floris.farm.set_yaw_angles(opt_yaw_angles, calculate_wake=True)

# Optimal power
power_optimal = np.sum([turbine.power for turbine in floris.farm.turbines])
print('Power increased by {}%'.format(100 * (power_optimal - power_initial) / power_initial))
