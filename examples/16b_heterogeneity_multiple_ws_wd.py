
import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


"""
This example showcases the heterogeneous inflow capabilities of FLORIS
when multiple wind speeds and direction are considered.
"""


# Define the speed ups of the heterogeneous inflow, and their locations.
# For the 2-dimensional case, this requires x and y locations.
# The speed ups are multipliers of the ambient wind speed.
speed_ups = [[2.0, 1.0, 2.0, 1.0]]
x_locs = [-300.0, -300.0, 2600.0, 2600.0]
y_locs = [ -300.0, 300.0, -300.0, 300.0]

# Initialize FLORIS with the given input.
# Note the heterogeneous inflow is defined in the input file.
fmodel = FlorisModel("inputs/gch_heterogeneous_inflow.yaml")

# Set shear to 0.0 to highlight the heterogeneous inflow
fmodel.set(
    wind_shear=0.0,
    wind_speeds=[8.0],
    wind_directions=[270.],
    turbulence_intensities=[0.06],
    layout_x=[0, 0],
    layout_y=[-299., 299.],
)
fmodel.run()
turbine_powers = fmodel.get_turbine_powers().flatten() / 1000.

# Show the initial results
print('------------------------------------------')
print('Given the speedups and turbine locations, ')
print(' the first turbine has an inflow wind speed')
print(' twice that of the second')
print(' Wind Speed = 8., Wind Direction = 270.')
print(f'T0: {turbine_powers[0]:.1f} kW')
print(f'T1: {turbine_powers[1]:.1f} kW')
print()

# If the number of conditions in the calculation changes, a new heterogeneous map
# must be provided.
speed_multipliers = [[2.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0]] # Expand to two wind conditions
heterogenous_inflow_config = {
    'speed_multipliers': speed_multipliers,
    'x': x_locs,
    'y': y_locs,
}
fmodel.set(
    wind_directions=[270.0, 275.0],
    wind_speeds=[8.0, 8.0],
    turbulence_intensities=[0.06, 0.06],
    heterogenous_inflow_config=heterogenous_inflow_config
)
fmodel.run()
turbine_powers = np.round(fmodel.get_turbine_powers() / 1000.)
print('With wind directions now set to 270 and 275 deg')
print(f'T0: {turbine_powers[:, 0].flatten()} kW')
print(f'T1: {turbine_powers[:, 1].flatten()} kW')

# # Uncomment if want to see example of error output
# # Note if we change wind directions to 3 without a matching change to het map we get an error
# print()
# print()
# print('~~ Now forcing an error by not matching wd and het_map')

# fmodel.set(wind_directions=[270, 275, 280], wind_speeds=3*[8.0])
# fmodel.run()
# turbine_powers = np.round(fmodel.get_turbine_powers() / 1000.)
