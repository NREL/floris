
import numpy as np

from floris import FlorisModel


# Initialize FLORIS with the given input file via FlorisModel.
# For basic usage, FlorisModel provides a simplified and expressive
# entry point to the simulation routines.
fmodel = FlorisModel("inputs/gch.yaml")

# Create a 4-turbine layouts
fmodel.set(layout_x=[0, 0., 500., 500.], layout_y=[0., 300., 0., 300.])

# Calculate wake
fmodel.run()

# Collect the wind speed at all the turbine points
u_points = fmodel.core.flow_field.u

print('U points is 1 findex x 4 turbines x 3 x 3 points (turbine_grid_points=3)')
print(u_points.shape)

print('turbine_average_velocities is 1 findex x 4 turbines')
print(fmodel.turbine_average_velocities)

# Show that one is equivalent to the other following averaging
print(
    'turbine_average_velocities is determined by taking the cube root of mean '
    'of the cubed value across the points '
)
print(f'turbine_average_velocities: {fmodel.turbine_average_velocities}')
print(f'Recomputed:       {np.cbrt(np.mean(u_points**3, axis=(2,3)))}')
