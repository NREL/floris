"""Example: Visualize flow by sweeping turbines

Demonstrate the use calculate_horizontal_plane_with_turbines

"""

import matplotlib.pyplot as plt
import numpy as np

import floris.flow_visualization as flowviz
from floris import FlorisModel


fmodel = FlorisModel("../inputs/gch.yaml")

viz_by_sweep = True

# # Some wake models may not yet have a visualization method included, for these cases can use
# # a slower version which scans a turbine model to produce the horizontal flow
x = np.array([-100, 0, 100, 200, 300, 400, 500, 600])
v = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

het_inflow_config = {
    "x": np.repeat(x, 2),
    "y": np.tile(np.array([-100.0, 100.0]), 8),
    #"z": np.array([90.0, 90.0, 90.0, 91.0]),
    "speed_multipliers": 1.0*np.ones((1, 16)),
    "u": 8.0*np.ones((1, v.shape[1]*2)),
    "v": np.repeat(v, 2, axis=1)
    #"v": np.array([[0.0, 0.0, 0.0, 0.0]]),
}

# Set a 2 turbine layout
fmodel.set(
    layout_x=[0, 300],
    layout_y=[0, 0],
    wind_directions=[270],
    wind_speeds=[8],
    turbulence_intensities=[0.06],
    heterogeneous_inflow_config=het_inflow_config,
)

if viz_by_sweep:
    horizontal_plane = flowviz.calculate_horizontal_plane_with_turbines(
        fmodel,
        x_resolution=20,
        y_resolution=10,
        x_bounds=(-100, 500),
        y_bounds=(-100, 100),
    )
    title = "Coarse turbine scan method"
else:
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        x_bounds=(-100, 500),
        y_bounds=(-100, 100),
    )
    title = "Standard flow visualization calculation"

fig, ax = plt.subplots(figsize=(10, 4))
flowviz.visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    label_contours=True,
    title=title
)


plt.show()
