"""
Example: Change operation model and compare power loss in yaw.

This example illustrates how to define different operational models and compares
the power loss resulting from yaw misalignment across these various models.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries


# Parameters
N = 101  # How many steps to cover yaw range in
yaw_max = 30  # Maximum yaw angle to test

# Set up the yaw angle sweep
yaw_angles = np.zeros((N, 1))
yaw_angles[:, 0] = np.linspace(-yaw_max, yaw_max, N)
print(yaw_angles.shape)


def evaluate_yawed_power(wsp: float, op_model: str) -> float:
    print(f"Evaluating model: {op_model}   wind speed: {wsp} m/s")

    # Grab model of FLORIS
    fmodel = FlorisModel("../inputs/gch.yaml")

    # Run N cases by setting up a TimeSeries (which is just several independent simulations)
    wind_directions = np.ones(N) * 270.0
    fmodel.set(
        wind_data=TimeSeries(
            wind_speeds=wsp,
            wind_directions=wind_directions,
            turbulence_intensities=0.06,
        )
    )

    yaw_angles = np.array(
        [(yaw, 0.0, 0.0) for yaw in np.linspace(-yaw_max, yaw_max, N)]
    )
    fmodel.set_operation_model(op_model)
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run()

    # Save the power output results in kW
    return fmodel.get_turbine_powers()[:, 0] / 1000


# Loop over the operational models and wind speeds to compare
op_models = ["simple", "cosine-loss", "mit-loss"]
wind_speeds = [11.0, 11.5, 15.0]
results = {}
for op_model, wsp in itertools.product(op_models, wind_speeds):

    # Save the power output results in kW
    results[(op_model, wsp)] = evaluate_yawed_power(wsp, op_model)
# Plot the results
fig, axes = plt.subplots(1, len(wind_speeds), sharey=True)

colors = ["C0", "k", "r"]
linestyles = ["solid", "dashed", "dotted"]
for wsp, ax in zip(wind_speeds, axes):
    ax.set_title(f"wsp: {wsp} m/s")
    ax.set_xlabel("Yaw angle [deg]")
    for op_model, c, ls in zip(op_models, colors, linestyles):

        upstream_yaw_angle = yaw_angles[:, 0]
        central_power = results[(op_model, wsp)][upstream_yaw_angle == 0]
        ax.plot(
            upstream_yaw_angle,
            results[(op_model, wsp)] / central_power,
            label=op_model,
            color=c,
            linestyle=ls,
        )

ax.grid(True)
ax.legend()
axes[0].set_xlabel("Yaw angle [deg]")
axes[0].set_ylabel("Normalized turbine power [-]")

plt.show()
