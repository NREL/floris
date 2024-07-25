import matplotlib.pyplot as plt
import numpy as np

import floris.layout_visualization as layoutviz
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


# Get a test fi (single turbine at 0,0)
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=[0, 0, 600, 600], layout_y=[0, -100, 0, -100])

fmodel.set(
    wind_directions=[270.0],
    wind_speeds=[8.0],
    turbulence_intensities=[0.06],
)
fmodel.run()
P_wo_het = fmodel.get_turbine_powers()/1e6

fmodel.set(
    heterogeneous_inflow_config={
        "wind_directions": [270.0, 280.0, 280.0, 290.0],
        "x": [-1000.0, -1000.0, 1000.0, 1000.0],
        "y": [-1000.0, 1000.0, -1000.0, 1000.0],
        "speed_multipliers":np.array([[1.0, 1.0, 1.0, 1.0]])#np.array([[1.0, 1.2, 1.2, 1.4]])
    },
)
fmodel.run()
P_w_het = fmodel.get_turbine_powers()/1e6

print("Difference (MW):", P_w_het - P_wo_het)

