
import numpy as np

from floris.tools import FlorisInterface


# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")

# Create a 4-turbine layouts
fi.set(layout_x=[0, 0., 500., 500.], layout_y=[0., 300., 0., 300.])

# Calculate wake
fi.run()

# Collect the wind speed at all the turbine points
u_points = fi.floris.flow_field.u

print('U points is 1 findex x 4 turbines x 3 x 3 points (turbine_grid_points=3)')
print(u_points.shape)

print('turbine_average_velocities is 1 findex x 4 turbines')
print(fi.turbine_average_velocities)

# Show that one is equivalent to the other following averaging
print(
    'turbine_average_velocities is determined by taking the cube root of mean '
    'of the cubed value across the points '
)
print(f'turbine_average_velocities: {fi.turbine_average_velocities}')
print(f'Recomputed:       {np.cbrt(np.mean(u_points**3, axis=(2,3)))}')
