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
 

import matplotlib.pyplot as plt
import floris.tools as wfct

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Show the current model parameters
print('All the model parameters and their current values:\n')
fi.show_model_parameters()
print('\n')

# Show the current model parameters with docstring info
print('All the model parameters, their current values, and docstrings:\n')
fi.show_model_parameters(verbose=True)
print('\n')

# Show a specific model parameter with its docstring
print('A specific model parameter, its current value, and its docstring:\n')
fi.show_model_parameters(params=['ka'], verbose=False)
print('\n')

# Get the current model parameters
model_params = fi.get_model_parameters()
print('The current model parameters:\n')
print(model_params)
print('\n')

# Set parameters on the current model
print('Set specific model parameters on the current wake model:\n')
params = {
    'Wake Velocity Parameters': {'alpha': 0.2},
    'Wake Deflection Parameters': {'alpha': 0.2},
    'Wake Turbulence Parameters': {'ti_constant': 1.0}
}
fi.set_model_parameters(params)
print('\n')

# Check that the parameters were changed
print('Observe that the requested paremeters changes have been made:\n')
model_params = fi.get_model_parameters()
print(model_params)
print('\n')
