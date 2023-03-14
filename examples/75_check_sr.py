# Copyright 2021 NREL

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

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.tools.visualization import plot_rotor_values, visualize_cut_plane


fi = FlorisInterface("inputs/gch.yaml")

# Set up simple 4 turbine layout
D = fi.floris.farm.rotor_diameters[0]
num_in_row = 4 # 20
fi.reinitialize(
    layout_x=[x*5.0*D for x in range(num_in_row)],
    layout_y=[0.0]*num_in_row,
    wind_speeds=[8.0],
    wind_directions=[270.0]
)

fi.calculate_wake()



max_ang = 50


yaw_opt = YawOptimizationSR(fi,
    minimum_yaw_angle=-max_ang,
    maximum_yaw_angle=max_ang
)


df_opt_gch = yaw_opt.optimize()



fi.calculate_wake(yaw_angles=df_opt_gch.loc[0, "yaw_angles_opt"][None, None, :])

# Look at the powers of each turbine
turbine_powers = fi.get_turbine_powers().flatten()


print("GCH optimal:", df_opt_gch.loc[0, "yaw_angles_opt"])
print(turbine_powers / 1e3)
print(np.sum(turbine_powers) / 1e3)

# Now sweep the angles of the third turbine and so how power changes
yaw_angle_sweep = np.linspace(-max_ang, max_ang, 30)
turbine_powers_sweep = np.zeros((len(yaw_angle_sweep), 4))
for i, yaw_angle in enumerate(yaw_angle_sweep):
    yaw_angles = df_opt_gch.loc[0, "yaw_angles_opt"].copy()
    yaw_angles[2] = yaw_angle
    fi.calculate_wake(yaw_angles=yaw_angles[None, None, :])
    turbine_powers_sweep[i, :] = fi.get_turbine_powers().flatten()

# plot the turbine powers
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(yaw_angle_sweep, turbine_powers_sweep[:, i] / 1e3, label=f"Turbine {i}")
ax.set_xlabel("Yaw angle (deg)")
ax.set_ylabel("Turbine power (kW)")
ax.legend()


# Plot the powers over the sum
fig, ax = plt.subplots()
ax.plot(yaw_angle_sweep, np.sum(turbine_powers_sweep, axis=1) / 1e3,'s-', label="Total", color="k")
ax.set_xlabel("Yaw angle of 3rd turbine (deg)")
ax.set_ylabel("Total power (kW)")
ax.legend()
ax.grid(True)
plt.show()
