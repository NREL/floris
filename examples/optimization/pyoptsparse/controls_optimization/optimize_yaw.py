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
 

import os
import floris.tools.optimization.pyoptsparse as opt
import floris.tools as wfct

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, '../../../example_input.json')
)

# Set turbine locations to 4 turbines in a rectangle
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 0, 6*D, 6*D]
layout_y = [0, 5*D, 0, 5*D]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

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
