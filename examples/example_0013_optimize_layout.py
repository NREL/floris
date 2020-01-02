# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
import numpy as np

# Instantiate the FLORIS object
fi = wfct.floris_interface.FlorisInterface("example_input.json")

# Set turbine locations to 3 turbines in a triangle
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [10, 10, 10+7*D]
layout_y = [200, 1000, 200]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Define the boundary for the wind farm
boundaries = [[2000., 4000.], [2000., 0.1], [0., 0.], [0., 2000.]]

# Generate random wind rose data
wd = np.arange(0., 360., 5.)
np.random.seed(1)
ws = 8.0 + np.random.randn(len(wd))*0.5
freq = np.abs(np.sort(np.random.randn(len(wd))))
freq = freq/freq.sum()

# Set optimization options
opt_options = {'maxiter': 50, 'disp': True, 'iprint': 2, 'ftol': 1e-8}

# Compute initial AEP for optimization normalization
AEP_initial = fi.get_farm_AEP(wd, ws, freq)

# Instantiate the layout otpimization object
layout_opt = wfct.optimization.LayoutOptimization(
                        fi=fi,
                        boundaries=boundaries,
                        wd=wd,
                        ws=ws,
                        freq=freq,
                        AEP_initial=AEP_initial,
                        opt_options=opt_options
)

# Perform layout optimization
layout_results = layout_opt.optimize()

print('=====================================================')
print('Layout coordinates: ')
for i in range(len(layout_results[0])):
    print('Turbine', i, ': \tx = ', '{:.1f}'.format(layout_results[0][i]), \
          '\ty = ', '{:.1f}'.format(layout_results[1][i]))

# Calculate new AEP results
fi.reinitialize_flow_field(layout_array=(layout_results[0], layout_results[1]))
AEP_optimized = fi.get_farm_AEP(wd, ws, freq)

print('=====================================================')
print('Total AEP Gain = %.1f%%' %
      (100.*(AEP_optimized - AEP_initial)/AEP_initial))
print('=====================================================')

# Plot the new layout vs the old layout
layout_opt.plot_layout_opt_results()
plt.show()