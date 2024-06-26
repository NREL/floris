from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel


# Get a test fi (single turbine at 0,0)
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=[0], layout_y=[0])

# Directly downstream at 270 degrees
option = False
if not option:
    sample_x = [500.0]
    sample_y = [0.0]
    sample_z = [90.0]

    # Sweep across wind directions
    wd_array = np.arange(180, 360, 1)
    ws_array = 8.0 * np.ones_like(wd_array)
    ti_array = 0.06 * np.ones_like(wd_array)
    fmodel.set(wind_directions=wd_array, wind_speeds=ws_array, turbulence_intensities=ti_array)

    # Standard case; expect minimum to be at 270 degrees
    u_at_points_0 = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z)

    # Now, apply bulk wind direction heterogeneity to the flow
    fmodel.set(
        heterogeneous_inflow_config={
            "bulk_wd_change": [[0.0, 10.0]]*180, # -10 degree change, CW
            "bulk_wd_x": [[0, 500.0]]*180
        }
    ) # TODO: Build something that checks the dimensions of the inputs here
    u_at_points_1 = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z)

    fmodel.set(
        heterogeneous_inflow_config={
            "bulk_wd_change": [[0.0, -10.0]]*180, # -10 degree change, CW
            "bulk_wd_x": [[0, 500.0]]*180
        }
    ) # TODO: Build something that checks the dimensions of the inputs here
    u_at_points_2 = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z)


    fig, ax = plt.subplots(1,1)
    ax.plot(wd_array, u_at_points_0, label="Standard", color="black")
    ax.plot(wd_array, u_at_points_1, label="Shifted 10")
    ax.plot(wd_array, u_at_points_2, label="Shifted -10")

else:
    ### Let's try something else
    sample_x = [500.0]*100
    sample_y = np.linspace(-500, 500, 100)
    sample_z = [90.0]*100

    fmodel.set(
        wind_directions=[270.0],
        wind_speeds=[8.0],
        turbulence_intensities=[0.06],
        heterogeneous_inflow_config={}
    )
    u_at_points_00 = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z).flatten()

    fmodel.set(
        heterogeneous_inflow_config={
            "bulk_wd_change": [[0.0, -30.0]], # -10 degree change, CW
            "bulk_wd_x": [[0, 500.0]]
        },
    )
    u_at_points_01 = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z).flatten()

    fig, ax = plt.subplots(1,1)
    ax.plot(sample_y, u_at_points_00, label="Standard", color="black")
    ax.plot(sample_y, u_at_points_01, label="Shifted 10")

plt.show()


