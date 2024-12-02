"""Example: Compare yaw loss
This example shows demonstrates how the Controller-dependent operation model (developed at TUM)
alters how a turbine loses power to yaw compared to the standard cosine loss model.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel


# Parameters
N  = 101 # How many steps to cover yaw range in
yaw_max = 30 # Maximum yaw to test

# Set up the yaw angle sweep
yaw_angles = np.zeros((N,1))
yaw_angles[:,0] = np.linspace(-yaw_max, yaw_max, N)

# Create the FLORIS model
fmodel = FlorisModel("../inputs/gch.yaml")

# Initialize to a simple 1 turbine case with n_findex = N
fmodel.set(
    layout_x=[0],
    layout_y=[0],
    wind_directions=270 * np.ones(N),
    wind_speeds=8 * np.ones(N),
    turbulence_intensities=0.06 * np.ones(N),
    yaw_angles=yaw_angles,
)

# Loop over the operational models to compare
op_models = ["cosine-loss", "controller-dependent"]
results = {}

for op_model in op_models:

    print(f"Evaluating model: {op_model}")
    fmodel.set_operation_model(op_model)

    fmodel.run()
    results[op_model] = fmodel.get_turbine_powers().squeeze()

# Plot the results
fig, ax = plt.subplots()
colors = ["C0", "k"]
linestyles = ["solid", "dashed"]
for key, c, ls in zip(results, colors, linestyles):
    central_power = results[key][yaw_angles.squeeze() == 0]
    ax.plot(yaw_angles.squeeze(), results[key]/central_power, label=key, color=c, linestyle=ls)

ax.grid(True)
ax.legend()
ax.set_xlabel("Yaw angle [deg]")
ax.set_ylabel("Normalized turbine power [deg]")

plt.show()
