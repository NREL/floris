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
 
# This example optimization takes a 1x5 array of turbines at an initial spacing
# of 9 rotor diameters and works to compress that spacing in the streamwise (x)
# direction, while working to maintain the original energy output by leveraging
# wake steering. It is meant to be an illustrative example of some of the
# benefits of wake steering.

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.power_density_1D \
    import PowerDensityOptimization1D
import numpy as np
import os

# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, '../../../example_input.json')
)

# Set turbine locations to 5 turbines in a line
D = fi.floris.farm.turbines[0].rotor_diameter
spacing = 9*D
layout_x = [0, spacing, 2*spacing, 3*spacing, 4*spacing]
layout_y = [0, 0, 0, 0, 0]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Wind inputs
wd = [270]
ws = [8]
freq = [1]

# Set optimization options
opt_options = {'maxiter': 50, 'disp': True, 'iprint': 2, 'ftol': 1e-9}

# Compute initial AEP for optimization normalization
AEP_initial = fi.get_farm_AEP(wd, ws, freq)

# Set initial conditions for optimization (scaled between 0 and 1)
x0 = [0., 0.25, 0.5, 0.75, 1.0] + [0.25, 0., 0., 0., 0.]

# Instantiate the layout otpimization object
powdens_opt = PowerDensityOptimization1D(
    fi=fi,
    wd=wd,
    ws=ws,
    freq=freq,
    AEP_initial=AEP_initial,
    x0=x0,
    opt_options=opt_options
)

# Perform layout optimization
powdens_results = powdens_opt.optimize()

# print(layout_x)
# print(powdens_results)

print('=====================================================')
print('Layout coordinates: ')
for i in range(len(powdens_results[0])):
    print('Turbine', i, ': \tx = ', '{:.1f}'.format(powdens_results[0][i]), \
          '\ty = ', '{:.1f}'.format(layout_y[i]))

# Calculate new AEP results
fi.reinitialize_flow_field(
    layout_array=(powdens_results[0],
    powdens_results[1])
)
AEP_optimized = fi.get_farm_AEP(wd, ws, freq)

print('=====================================================')
print('AEP Ratio = %.1f%%' %
      (100.*AEP_optimized/AEP_initial))
print('=====================================================')

# Plot the new layout vs the old layout
powdens_opt.plot_layout_opt_results()
plt.show()
