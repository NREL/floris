import numpy as np
import matplotlib.pyplot as plt

import floris.layout_visualization as layoutviz
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


# Get a test fi (single turbine at 0,0)
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=[0, 600], layout_y=[0, 0])

# Directly downstream at 270 degrees
### Let's try something else
sample_x = [500.0]*100
sample_y = np.linspace(-500, 500, 100)
sample_z = [90.0]*100

fmodel.set(
    wind_directions=[270.0],
    wind_speeds=[8.0],
    turbulence_intensities=[0.06],
    heterogeneous_inflow_config={
        "bulk_wd_change": [[0.0, -15.0]], # -10 degree change, CW
        "bulk_wd_x": [[0, 2000.0]]
    },
)
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
)

# Plot the flow field with rotors
fig, ax = plt.subplots()
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    label_contours=False,
    title="Horizontal Flow with Turbine Rotors and labels",
)

# Plot the turbine rotors
layoutviz.plot_turbine_rotors(fmodel, ax=ax)

plt.show()


