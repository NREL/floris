# Copyright 2023 NREL

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
import numpy as np
import yaml

from floris import FlorisModel


"""
This example demonstrates the use of the sample_flow_at_points method of
FlorisInterface. sample_flow_at_points extracts the wind speed
information at user-specified locations in the flow.

Specifically, this example returns the wind speed at a single x, y
location and four different heights over a sweep of wind directions.
This mimics the wind speed measurements of a met mast across all
wind directions (at a fixed free stream wind speed).

Try different values for met_mast_option to vary the location of the
met mast within the two-turbine farm.
"""

# User options
# FLORIS model to use (limited to Gauss/GCH, Jensen, and empirical Gauss)
floris_model = "emgauss" # Try "gch", "jensen", "emgauss"
# Option to try different met mast locations
met_mast_option = 0 # Try 0, 1, 2, 3

# Instantiate FLORIS model
fmodel = FlorisModel("inputs/"+floris_model+"_iea_15MW.yaml")

with open(str(
    fmodel.core.as_dict()["farm"]["turbine_library_path"] /
    (fmodel.core.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "helix"

fmodel.set(layout_x=[0], layout_y=[0], turbine_type=['iea_15mw'])
D = 240

fig, ax = plt.subplots(2,1)


# Set the wind direction to run 360 degrees
N = 4
helix_amplitudes = np.array([0, 1, 2.5, 4]).reshape(1, N).T
fmodel.set(wind_directions=270 * np.ones(N), wind_speeds=7.5 * np.ones(N), turbulence_intensities=0.06*np.ones(N),helix_amplitudes=helix_amplitudes)
fmodel.run()

# Simulate a met mast in between the turbines
x = np.linspace(-1*D, 10*D, 111)
y = np.linspace(-0.5*D, 0.5*D, 25)
points_x, points_y = np.meshgrid(x, y)

points_x = points_x.flatten()
points_y = points_y.flatten()
points_z = 150*np.ones_like(points_x)

# Collect the points
fmodel.core.farm.hub_heights = [150]
u_at_points = fmodel.sample_flow_at_points(points_x, points_y, points_z)

vel_in_wake = np.average(np.reshape(u_at_points, (N, len(x), len(y)), 'F'), axis=2).T

amr_data = np.loadtxt("examples/inputs/wakerecovery_helix.csv", delimiter=",", dtype=float)

# Plot the velocities
for n in range(len(vel_in_wake[0,:])):
    ax[0].plot(x, vel_in_wake[:,n], color='C'+str(n))
    ax[0].plot(amr_data[:,0], amr_data[:,n+1], color='C'+str(n), linestyle='--',label='_nolegend_')
ax[0].grid()
ax[0].set_title('')
ax[0].legend(['Baseline','Helix, 1 deg', 'Helix, 2.5 deg','Helix, 4 deg'])

for n in range(len(vel_in_wake[0,:])-1):
    ax[1].plot(x, vel_in_wake[:,n+1]/vel_in_wake[:,0], color='C'+str(n+1))
    ax[1].plot(amr_data[:,0], amr_data[:,n+2]/amr_data[:,1], color='C'+str(n+1), linestyle='--')
ax[1].grid()
ax[1].legend(['Floris','AMR-Wind'])
ax[1].set_xlabel('Wind Direction (deg)')
ax[1].set_ylabel('Wind Speed (m/s)')

plt.show()
