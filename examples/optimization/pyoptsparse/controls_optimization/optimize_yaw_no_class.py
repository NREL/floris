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
 
import pyoptsparse
import os

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

def objective_function(varDict, **kwargs):
    # Parse the variable dictionary
    yaw = varDict['yaw']

    # Compute the objective function
    funcs = {}
    funcs['obj'] = -1*fi.get_farm_power_for_yaw_angle(yaw)/1e5

    fail = False
    return funcs, fail

# Setup the optimization problem
optProb = pyoptsparse.Optimization('yaw_opt', objective_function)

# Add the design variables to the optimization problem
optProb.addVarGroup('yaw', 4, 'c', lower=0, upper= 20, value=2.)

# Add the objective to the optimization problem
optProb.addObj('obj')

# Setup the optimization solver
# Note: pyOptSparse has other solvers available; some may require additional
#   licenses/installation. See https://github.com/mdolab/pyoptsparse for more
#   information. When ready, they can be invoked by changing 'SLSQP' to the
#   solver name, for example: 'opt = pyoptsparse.SNOPT(fi=fi)'.
opt = pyoptsparse.SLSQP(fi=fi)

# Run the optimization with finite-differencing
solution = opt(optProb, sens='FD')
print(solution)
