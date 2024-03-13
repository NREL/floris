
import matplotlib.pyplot as plt
import numpy as np

import floris.flow_visualization as flowviz
from floris import FlorisModel


"""
This example initializes the FLORIS software, and then uses internal
functions to run a simulation and plot the results. In this case,
we are plotting three slices of the resulting flow field:
1. Horizontal slice parallel to the ground and located at the hub height
2. Vertical slice of parallel with the direction of the wind
3. Vertical slice parallel to to the turbine disc plane

Additionally, an alternative method of plotting a horizontal slice
is shown. Rather than calculating points in the domain behind a turbine,
this method adds an additional turbine to the farm and moves it to
locations throughout the farm while calculating the velocity at it's
rotor.
"""

# Initialize FLORIS with the given input file via FlorisModel.
# For basic usage, FlorisModel provides a simplified and expressive
# entry point to the simulation routines.
fmodel = FlorisModel("inputs/gch.yaml")

# The rotor plots show what is happening at each turbine, but we do not
# see what is happening between each turbine. For this, we use a
# grid that has points regularly distributed throughout the fluid domain.
# The FlorisModel contains functions for configuring the new grid,
# running the simulation, and generating plots of 2D slices of the
# flow field.

# Note this visualization grid created within the calculate_horizontal_plane function will be reset
# to what existed previously at the end of the function

# Using the FlorisModel functions, get 2D slices.
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
    yaw_angles=np.array([[25.,0.,0.]]),
)

y_plane = fmodel.calculate_y_plane(
    x_resolution=200,
    z_resolution=100,
    crossstream_dist=0.0,
    yaw_angles=np.array([[25.,0.,0.]]),
)
cross_plane = fmodel.calculate_cross_plane(
    y_resolution=100,
    z_resolution=100,
    downstream_dist=630.0,
    yaw_angles=np.array([[25.,0.,0.]]),
)

# Create the plots
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
flowviz.visualize_cut_plane(
    horizontal_plane,
    ax=ax_list[0],
    label_contours=True,
    title="Horizontal"
)
flowviz.visualize_cut_plane(
    y_plane,
    ax=ax_list[1],
    label_contours=True,
    title="Streamwise profile"
)
flowviz.visualize_cut_plane(
    cross_plane,
    ax=ax_list[2],
    label_contours=True,
    title="Spanwise profile"
)

# Some wake models may not yet have a visualization method included, for these cases can use
# a slower version which scans a turbine model to produce the horizontal flow
horizontal_plane_scan_turbine = flowviz.calculate_horizontal_plane_with_turbines(
    fmodel,
    x_resolution=20,
    y_resolution=10,
    yaw_angles=np.array([[25.,0.,0.]]),
)

fig, ax = plt.subplots()
flowviz.visualize_cut_plane(
    horizontal_plane_scan_turbine,
    ax=ax,
    label_contours=True,
    title="Horizontal (coarse turbine scan method)",
)

# FLORIS further includes visualization methods for visualing the rotor plane of each
# Turbine in the simulation

# Run the wake calculation to get the turbine-turbine interfactions
# on the turbine grids
fmodel.run()

# Plot the values at each rotor
fig, axes, _ , _ = flowviz.plot_rotor_values(
    fmodel.core.flow_field.u,
    findex=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("Rotor Plane Visualization, Original Resolution")

# FLORIS supports multiple types of grids for capturing wind speed
# information. The current input file is configured with a square grid
# placed on each rotor plane with 9 points in a 3x3 layout. For visualization,
# this resolution can be increased.  Note this operation, unlike the
# calc_x_plane above operations does not automatically reset the grid to
# the initial status as definied by the input file

# Increase the resolution of points on each turbien plane
solver_settings = {
    "type": "turbine_grid",
    "turbine_grid_points": 10
}
fmodel.set(solver_settings=solver_settings)

# Run the wake calculation to get the turbine-turbine interfactions
# on the turbine grids
fmodel.run()

# Plot the values at each rotor
fig, axes, _ , _ = flowviz.plot_rotor_values(
    fmodel.core.flow_field.u,
    findex=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("Rotor Plane Visualization, 10x10 Resolution")

# Show plots
plt.show()

# Note if the user doesn't import matplotlib.pyplot as plt, the user can
# use the following to show the plots:
# flowviz.show()
