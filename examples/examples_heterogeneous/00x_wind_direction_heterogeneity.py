import matplotlib.pyplot as plt
import numpy as np

import floris.layout_visualization as layoutviz
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


visualize = True

# Get a test fi (single turbine at 0,0)
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=[0, 0, 600, 600], layout_y=[0, -300, 0, -300])

fmodel.set(
    wind_directions=[270.0],
    wind_speeds=[8.0],
    turbulence_intensities=[0.06],
    wind_shear=0.0
)
fmodel.run()
P_wo_het = fmodel.get_turbine_powers()/1e6

het_inflow_config = {
    "x": np.array([-1000.0, -1000.0, 1000.0, 1000.0]),
    "y": np.array([-500.0, 500.0, -500.0, 500.0]),
    "speed_multipliers": np.array([[1.0, 1.0, 1.0, 1.0]]),
    "u": np.array([[8.0, 8.0, 8.0, 8.0]]),
    "v": np.array([[-2.0, 0.0, -2.0, 0.0]]),
    #"v": np.array([[0.0, 0.0, 0.0, 0.0]]),
}

fmodel.set(heterogeneous_inflow_config=het_inflow_config)
fmodel.run()
P_w_het = fmodel.get_turbine_powers()/1e6

print("Difference (MW):", P_w_het - P_wo_het)

if visualize:
    fig, ax = plt.subplots(2, 1, figsize=(4,8))
    fmodel.set(
        heterogeneous_inflow_config={
            "x": np.array([-1000.0, -1000.0, 1000.0, 1000.0]),
            "y": np.array([-1000.0, 1000.0, -1000.0, 1000.0]),
            "speed_multipliers":np.array([[1.0, 1.0, 1.0, 1.0]])
        }
    )
    fmodel.run()
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        x_bounds=(-200, 1000),
        y_bounds=(-500, 500),
    )
    visualize_cut_plane(
        horizontal_plane,
        ax=ax[0],
        label_contours=False,
        title="Without WD het"
    )

    fmodel.set(heterogeneous_inflow_config=het_inflow_config)
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        x_bounds=(-200, 1000),
        y_bounds=(-500, 500),
    )
    #import ipdb; ipdb.set_trace()
    visualize_cut_plane(
        horizontal_plane,
        ax=ax[1],
        label_contours=False,
        title="With WD het"
    )

    plt.show()
