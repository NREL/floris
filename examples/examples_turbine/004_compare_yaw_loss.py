"""
Example: Change operation model and compare power loss in yaw.

This example illustrates how to define different operational models and compares
the power loss resulting from yaw misalignment across these various models.
"""

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


# Loop over the operational models to compare
op_models = ["cosine-loss", "mit-loss"]
results = {}

for op_model in op_models:

    print(f"Evaluating model: {op_model}")

    # Grab model of FLORIS
    fmodel = FlorisModel("../inputs/gch.yaml")

    # Run N cases by setting up a TimeSeries (which is just several independent simulations)
    wind_directions = np.ones(N) * 270.0
    fmodel.set(
        wind_data=TimeSeries(
            wind_speeds=11.5,
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
    results[op_model] = fmodel.get_turbine_powers()[:, 0] / 1000

# Plot the results
fig, ax = plt.subplots()

colors = ["C0", "k", "r"]
linestyles = ["solid", "dashed", "dotted"]
for key, c, ls in zip(results, colors, linestyles):
    upstream_yaw_angle = yaw_angles[:, 0]
    central_power = results[key][upstream_yaw_angle == 0]
    ax.plot(
        upstream_yaw_angle,
        results[key] / central_power,
        label=key,
        color=c,
        linestyle=ls,
    )

ax.grid(True)
ax.legend()
ax.set_xlabel("Yaw angle [deg]")
ax.set_ylabel("Normalized turbine power [deg]")

plt.show()
