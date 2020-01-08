# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import floris.tools.optimization as opt
import floris.tools as wfct

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface('../example_input.json')

# Define wind speed and direction
ws = [8]
wd = [270]

# Set bounds on yaw offsets
minimum_yaw_angle = 0.
maximum_yaw_angle = 30.

model = opt.yaw.Yaw(fi, minimum_yaw_angle, maximum_yaw_angle, wdir=wd,
                                          wspd=ws)

tmp = opt.optimization.Optimization(model=model, solver='SLSQP')

sol = tmp.optimize()

# Display results
print(sol)

model.print_power_gain(sol)

model.plot_yaw_opt_results(sol)