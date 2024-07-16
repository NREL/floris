"""Example: Visualizing flow with wake steering

This example demonstrates how FLORIS models the effect of yaw misalignment on the wakes
of wind turbines.
"""

import matplotlib.pyplot as plt

from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


# Initialize the FLORIS model with a two-turbine layout
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=[0, 1000.0], layout_y=[0.0, 0.0])

fig, ax = plt.subplots(2,1)

# First run the aligned case and plot
fmodel.set(
    wind_speeds=[8.0],
    wind_directions=[270.0],
    turbulence_intensities=[0.06],
    yaw_angles=[[0.0, 0.0]]
)
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
)
visualize_cut_plane(
    horizontal_plane,
    ax=ax[0],
    label_contours=False,
    title="Aligned case (no wake steering)",
)

# Then, apply a 20 degree yaw misalignment to the upstream turbine and rerun
fmodel.set(
    yaw_angles=[[20.0, 0.0]]
)
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
)
visualize_cut_plane(
    horizontal_plane,
    ax=ax[1],
    label_contours=False,
    title="20 degree misaligned case (wake steering)",
)

plt.show()