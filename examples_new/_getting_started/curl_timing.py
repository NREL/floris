# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
from floris.utilities import Vec3
import time

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("../example_input_full.json")
fi.floris.farm.set_wake_model('curl')

np.random.seed(4365542)

# The rotor diameter
D = 126

# The turbine height
h = 90

# The domain sizes [m]
Lx = 40 * D
Ly = 20 * D
Lz = 300

# The location of the turbines
N = 36
N_x = 12 # The number of discrete point
N_y = 10 # The number of discrete point
layout_x = np.random.randint(1, N_x, size=N) * Lx / N_x * 0.9 + Lx * 0.1/2 #+ D/2 * np.random.rand(N)
layout_y = np.random.randint(1, N_y, size=N) * Ly / N_y * 0.9 + Ly * 0.1/2 #+ D/2 * np.random.rand(N)

print(layout_x)
print(layout_y)
print(len(layout_x))
print(len(layout_y))


# The length of the domain
x_min = np.amin(np.array(layout_x))
x_max = np.amax(np.array(layout_x))

y_min = np.amin(np.array(layout_y))
y_max = np.amax(np.array(layout_y))

dx = 10
dy = 20
dz = 20

i = 3
Nx = int(Lx/dx) * i / 2
Ny = int(Ly/dy) * i / 2
Nz = int(Lz/dz) * i / 2



params = fi.get_model_parameters()
print(params)
print(Nx, Ny, Nz)

# params['Wake Velocity Parameters']['model_grid_resolution'] = Vec3(Nx, Ny, Nz)
# print(params)

# lkj


fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])

print(len(fi.floris.farm.turbines))
# lkj
# Calculate wake
start = time.time()
fi.calculate_wake()
end = time.time()

print(str(Nx*Ny*Nz) + ' ' + str(end - start))

# Get horizontal plane at default height (hub-height)
# hor_plane = fi.get_hor_plane()

# # Plot and show
# fig, ax = plt.subplots()
# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
# plt.show()
