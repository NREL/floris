"""Example: Eddy viscosity model
This example demonstrates the wake profile of the eddy viscosity wake model,
presented originally by Ainslie (1988) and updated by Gunn (2019).
Links:
- Ainslie (1988): https://doi.org/10.1016/0167-6105(88)90037-2
- Gunn (2019): https://dx.doi.org/10.1088/1742-6596/1222/1/012003
"""

import matplotlib.pyplot as plt
import numpy as np

import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz
from floris import FlorisModel

# Instantiate FLORIS model
fmodel = FlorisModel("../inputs/ev.yaml")

## Plot the flow velocity profiles
# Set up a two-turbine farm
D = 126
fmodel.set(layout_x=[0, 500, 1000], layout_y=[0, 0, 0])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 4)
ax.scatter(fmodel.layout_x, fmodel.layout_y, color="red", label="Turbine location")

# Set a single wind condition of westerly wind
wd_array = np.array([270])
ws_array = 8.0 * np.ones_like(wd_array)
ti_array = 0.06 * np.ones_like(wd_array)
fmodel.set(wind_directions=wd_array, wind_speeds=ws_array, turbulence_intensities=ti_array)

# Create points to sample the flow in the turbines' wakes, both at the centerline and
# across the wake's width
x_locs = np.linspace(0, 2000, 400)
y_locs = np.linspace(-100, 100, 40)
points_x, points_y = np.meshgrid(x_locs, y_locs)
points_x = points_x.flatten()
points_y = points_y.flatten()
points_z = 90 * np.ones_like(points_x)

# Collect the points
u_at_points = fmodel.sample_flow_at_points(points_x, points_y, points_z)
u_at_points = u_at_points.reshape((len(y_locs), len(x_locs), 1))

# Plot the flow velocities
for y_idx, y in enumerate(y_locs):
    a = 1-np.abs(y/100)
    ax.plot(x_locs, u_at_points[y_idx, :, 0].flatten(), color="black", alpha=a)
ax.grid()
ax.legend()
ax.set_xlabel("x location [m]")
ax.set_ylabel("Wind Speed [m/s]")

## Visualize the flow in aligned and slightly misaligned conditions for a 9-turbine farm
D = 126.0
x_locs = np.array([0, 5*D, 10*D])
y_locs = x_locs
points_x, points_y = np.meshgrid(x_locs, y_locs)
fmodel.set(
    layout_x = points_x.flatten(),
    layout_y = points_y.flatten()
)

# Aligned
ax = layoutviz.plot_turbine_rotors(fmodel)
layoutviz.plot_turbine_labels(fmodel, ax)

fmodel.set(
    wind_speeds=[8.0, 8.0],
    wind_directions=[270.0, 270.0+15.0],
    turbulence_intensities=[0.06, 0.06]
)
fmodel.run()
cut_plane = fmodel.calculate_horizontal_plane(height=90, findex_for_viz=0)

flowviz.visualize_cut_plane(cut_plane, ax=ax)
ax.set_title("Aligned flow")

# Plot and print the power output of the turbines in each row
fig, ax = plt.subplots(1,1)
np.set_printoptions(formatter={"float": "{0:0.3f}".format})
print("Aligned case:")
for i in range(3):
    idxs = [3*i, 3*i+1, 3*i+2]
    ax.scatter([0, 1, 2], fmodel.get_turbine_powers()[0, idxs]/1e6, label="Column {0}".format(i))
    print(idxs, " -- ", fmodel.get_turbine_powers()[0, idxs]/1e6)
ax.grid()
ax.legend()
ax.set_xlabel("Turbine in column")
ax.set_ylabel("Power [MW]")
ax.set_title("Aligned case")
ax.set_ylim([0.5, 1.8])

# Misaligned
ax = layoutviz.plot_turbine_rotors(fmodel, yaw_angles=(-15.0)*np.ones(9))
layoutviz.plot_turbine_labels(fmodel, ax)

cut_plane = fmodel.calculate_horizontal_plane(height=90, findex_for_viz=1)
flowviz.visualize_cut_plane(cut_plane, ax=ax)
ax.set_title("Misaligned flow")

fig, ax = plt.subplots(1,1)
print("\nMisaligned case:")
for i in range(3):
    idxs = [3*i, 3*i+1, 3*i+2]
    ax.scatter([0, 1, 2], fmodel.get_turbine_powers()[1, idxs]/1e6, label="Column {0}".format(i))
    print(idxs, " -- ", fmodel.get_turbine_powers()[1, idxs]/1e6)
ax.grid()
ax.legend()
ax.set_xlabel("Turbine in column")
ax.set_ylabel("Power [MW]")
ax.set_title("Misaligned case")
ax.set_ylim([0.5, 1.8])

plt.show()
