import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel
import floris.layout_visualization as layoutviz
import floris.flow_visualization as flowviz

# User options
# FLORIS model to use (limited to Gauss/GCH, Jensen, and empirical Gauss)
floris_model = "ev"  # Try "gch", "jensen", "emgauss"

# Instantiate FLORIS model
fmodel = FlorisModel("../inputs/" + floris_model + ".yaml")

# Set up a two-turbine farm
D = 126
fmodel.set(layout_x=[0, 500, 1000], layout_y=[0, 0, 0])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 4)
ax.scatter(fmodel.layout_x, fmodel.layout_y, color="red", label="Turbine location")

# Set the wind direction to run 360 degrees
wd_array = np.array([270])
ws_array = 8.0 * np.ones_like(wd_array)
ti_array = 0.06 * np.ones_like(wd_array)
fmodel.set(wind_directions=wd_array, wind_speeds=ws_array, turbulence_intensities=ti_array)

# Simulate a met mast in between the turbines
x_locs = np.linspace(0, 2000, 400)
y_locs = np.linspace(-100, 100, 40)
points_x, points_y = np.meshgrid(x_locs, y_locs)
points_x = points_x.flatten()
points_y = points_y.flatten()
points_z = 90 * np.ones_like(points_x)

# Collect the points
u_at_points = fmodel.sample_flow_at_points(points_x, points_y, points_z)
u_at_points = u_at_points.reshape((len(y_locs), len(x_locs), 1))

# Plot the velocities
for y_idx, y in enumerate(y_locs):
    a = 1-np.abs(y/100)
    ax.plot(x_locs, u_at_points[y_idx, :, 0].flatten(), color="black", alpha=a)
ax.grid()
ax.legend()
ax.set_xlabel("x location [m]")
ax.set_ylabel("Wind Speed [m/s]")

###
D = 126.0
x_locs = np.array([0, 5*D, 10*D])
y_locs = x_locs
points_x, points_y = np.meshgrid(x_locs, y_locs)
fmodel.set(
    layout_x = points_x.flatten(),
    layout_y = points_y.flatten()
)

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

fig, ax = plt.subplots(1,1)
for i in range(3):
    idxs = [3*i, 3*i+1, 3*i+2]
    ax.scatter([0, 1, 2], fmodel.get_turbine_powers()[0, idxs]/1e6, label="Column {0}".format(i))
    print(idxs, " -- ", fmodel.get_turbine_powers()[0, idxs]/1e6)
ax.grid()
ax.legend()
ax.set_xlabel("Turbine in column")
ax.set_ylabel("Power [MW]")

####
ax = layoutviz.plot_turbine_rotors(fmodel, yaw_angles=(-15.0)*np.ones(9))
layoutviz.plot_turbine_labels(fmodel, ax)

cut_plane = fmodel.calculate_horizontal_plane(height=90, findex_for_viz=1)
flowviz.visualize_cut_plane(cut_plane, ax=ax)

fig, ax = plt.subplots(1,1)
for i in range(3):
    idxs = [3*i, 3*i+1, 3*i+2]
    ax.scatter([0, 1, 2], fmodel.get_turbine_powers()[0, idxs]/1e6, label="Column {0}".format(i))
    print(idxs, " -- ", fmodel.get_turbine_powers()[0, idxs]/1e6)
ax.grid()
ax.legend()
ax.set_xlabel("Turbine in column")
ax.set_ylabel("Power [MW]")

plt.show()
