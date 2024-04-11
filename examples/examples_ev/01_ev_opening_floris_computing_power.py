"""Example 1: Opening FLORIS and Computing Power

This first example illustrates several of the key concepts in FLORIS. It:

  1) Initializing FLORIS
  2) Changing the wind farm layout
  3) Changing the incoming wind speed, wind direction and turbulence intensity
  4) Running the FLORIS simulation
  5) Getting the power output of the turbines

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""


import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel


# Initialize FLORIS with the given input file.
# The Floris class is the entry point for most usage.
fmodel = FlorisModel("../inputs/ev.yaml")

# Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
fmodel.set(layout_x=[0, 500.0, 1000.0, 1500.0, 2000.0], layout_y=[0.0, 0.0, 0.0, 0.0, 0.0])

fmodel.set(
    wind_directions=np.array([270.0, 270.0, 270.0, 270.0]),
    wind_speeds=[8.0, 10.0, 12.0, 14.0],
    turbulence_intensities=np.array([0.06, 0.06, 0.06, 0.06])
)

# After the set method, the run method is called to perform the simulation
fmodel.run()

# There are functions to get either the power of each turbine, or the farm power
turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

print("Turbines:", turbine_powers)

print("Farm:", farm_power)


fig, ax = plt.subplots(1,1)
for i, ws in enumerate([8.0, 10.0, 12.0, 14.0]):
    ax.scatter(range(5), turbine_powers[i,:], label=f"Wind Speed: {ws}")
ax.legend()
ax.grid()
ax.set_xlabel("Turbine in row")
ax.set_ylabel("power [kW]")

plt.show()
