"""Example: Visualize flow by sweeping turbines

Demonstrate the use calculate_horizontal_plane_with_turbines

"""

import matplotlib.pyplot as plt
import numpy as np

import floris.flow_visualization as flowviz
from floris import FlorisModel


fmodel = FlorisModel("../inputs/gch.yaml")

# # Some wake models may not yet have a visualization method included, for these cases can use
# # a slower version which scans a turbine model to produce the horizontal flow
x = np.linspace(-100, 1000, 8)
v = np.linspace(0, 7, 8).reshape(1, 8)

het_inflow_config = {
    "x": np.repeat(x, 2),
    "y": np.tile(np.array([-200.0, 200.0]), 8),
    #"z": np.array([90.0, 90.0, 90.0, 91.0]),
    "speed_multipliers": 1.0*np.ones((1, 16)),
    "u": 8.0*np.ones((1, v.shape[1]*2)),
    "v": np.repeat(v, 2, axis=1)
    #"v": np.array([[0.0, 0.0, 0.0, 0.0]]),
}

# Set a 2 turbine layout
fmodel.set(
    layout_x=[0, 500, 0, 500],
    layout_y=[0, 0, -100, -100],
    wind_directions=[270],
    wind_speeds=[8],
    turbulence_intensities=[0.06],
    heterogeneous_inflow_config=het_inflow_config,
)

horizontal_plane_scan_turbine = flowviz.calculate_horizontal_plane_with_turbines(
    fmodel,
    x_resolution=20,
    y_resolution=10,
    x_bounds=(-100, 1000),
    y_bounds=(-200, 200),
)

fig, ax = plt.subplots(figsize=(10, 4))
flowviz.visualize_cut_plane(
    horizontal_plane_scan_turbine,
    ax=ax,
    label_contours=True,
    title="Horizontal (coarse turbine scan method)",
)


plt.show()
