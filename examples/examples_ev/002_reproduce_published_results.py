"""Example: Reproduce published eddy viscosity results
This example attempts to reproduce the results of Ainslie (1988) and Gunn (2019)
using the FLORIS implementation of the eddy viscosity model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import FlorisModel, TimeSeries
from floris.turbine_library import build_cosine_loss_turbine_dict

# Build a constant CT turbine model for use in comparisons (not realistic)
D = 120.0 # rotor diameter [m]
HH = 100.0 # hub height [m]
u_0 = 8.0 # wind speed [m/s]

# Load the EV model
fmodel = FlorisModel("../inputs/ev.yaml")

## First, reproduce results from Ainslie (1988)

# TODO: set up model parameters to match Ainslie (if possible)
fmodel.set_param(["wake", "wake_velocity_parameters", "eddy_viscosity", "filter_cutoff_D"], 2.5)

# Generate figure to plot on
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(4, 4)

for C_T, ls in zip([0.5, 0.7, 0.9], ["-", "--", ":"]):
    const_CT_turb = build_cosine_loss_turbine_dict(
        turbine_data_dict={
            "wind_speed":[0.0, 30.0],
            "power":[0.0, 1.0], # Not realistic but won't be used here
            "thrust_coefficient":[C_T, C_T]
        },
        turbine_name="ConstantCT",
        rotor_diameter=D,
        hub_height=HH,
        ref_tilt=0.0,
    )

    # Load the EV model and set a constant CT turbine
    fmodel.set(
        layout_x=[0],
        layout_y=[0],
        turbine_type=[const_CT_turb],
        wind_speeds=[u_0],
        wind_directions=[270],
        turbulence_intensities=[0.14],
        wind_shear=0.0
    )

    points_x = np.linspace(2*D, 10*D, 1000)
    points_y = np.zeros_like(points_x)
    points_z = HH * np.ones_like(points_x)

    u_at_points = fmodel.sample_flow_at_points(points_x, points_y, points_z)

    # Plot results (not different axis scales in Ainslie)
    ax.plot(
        points_x/D, 1-u_at_points[0, :]/u_0, color="black", linestyle=ls, label=rf"$C_T$ = {C_T}"
    )

ax.set_xlabel("Downstream distance [D]")
ax.set_ylabel("Centerline velocity deficit [-]")
ax.set_ylim([0, 1])
ax.legend()
ax.grid()

plt.show()