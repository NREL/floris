"""Example: Extract wind speed at points
This example demonstrates the use of the sample_flow_at_points method of
FlorisModel. sample_flow_at_points extracts the wind speed
information at user-specified locations in the flow.

Specifically, this example returns the wind speed at a single x, y
location and four different heights over a sweep of wind directions.
This mimics the wind speed measurements of a met mast across all
wind directions (at a fixed free stream wind speed).

Try different values for met_mast_option to vary the location of the
met mast within the two-turbine farm.
"""


import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel


# User options
# FLORIS model to use (limited to Gauss/GCH, Jensen, and empirical Gauss)
floris_model = "ev"  # Try "gch", "jensen", "emgauss"

# Instantiate FLORIS model
fmodel = FlorisModel("inputs/" + floris_model + ".yaml")

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

plt.show()
