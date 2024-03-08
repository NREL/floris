
import numpy as np

from floris import FlorisModel


"""
This example creates a FLORIS instance
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates multiple ws/wd simulations

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""

# Initialize FLORIS with the given input file.
# The Floris class is the entry point for most usage.
fmodel = FlorisModel("inputs/gch.yaml")

# Convert to a simple two turbine layout
fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

# Single wind speed and wind direction
print("\n========================= Single Wind Direction and Wind Speed =========================")

# Get the turbine powers assuming 1 wind direction and speed
# Set the yaw angles to 0 with 1 wind direction and speed
fmodel.set(wind_directions=[270.0], wind_speeds=[8.0], yaw_angles=np.zeros([1, 2]))

fmodel.run()

# Get the turbine powers
turbine_powers = fmodel.get_turbine_powers() / 1000.0

print("The turbine power matrix should be of dimensions 1 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

# Single wind speed and multiple wind directions
print("\n========================= Single Wind Direction and Multiple Wind Speeds ===============")

wind_speeds = np.array([8.0, 9.0, 10.0])
wind_directions = np.array([270.0, 270.0, 270.0])
turbulence_intensities = np.array([0.06, 0.06, 0.06])

# 3 wind directions/ speeds
fmodel.set(
    wind_speeds=wind_speeds,
    wind_directions=wind_directions,
    turbulence_intensities=turbulence_intensities,
    yaw_angles=np.zeros([3, 2])
)
fmodel.run()
turbine_powers = fmodel.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 3 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

# Multiple wind speeds and multiple wind directions
print("\n========================= Multiple Wind Directions and Multiple Wind Speeds ============")

# To consider each combination, this needs to be broadcast out in advance

wind_speeds = np.tile([8.0, 9.0, 10.0], 3)
wind_directions = np.repeat([260.0, 270.0, 280.0], 3)
turbulence_intensities = np.tile([0.06, 0.06, 0.06], 3)

fmodel.set(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    turbulence_intensities=turbulence_intensities,
    yaw_angles=np.zeros([9, 2])
)
fmodel.run()
turbine_powers = fmodel.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 9 WD/WS X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)
