"""Example: Create and supply a user-defined operation model

This example shows how to create a user-defined operation model and supply it to FLORIS.
It is based on an idealized actuator disk model that does not curtail (i.e. has no rated
wind speed).
"""

import numpy as np
from attrs import define, field
import matplotlib.pyplot as plt

from floris.type_dec import floris_float_type, NDArrayFloat
from floris.core.turbine.operation_models import BaseOperationModel
from floris.core.rotor_velocity import average_velocity

from floris import FlorisModel

# Declare the new operation model, inheriting from BaseOperationModel. The `@define` decorator from
# the `attrs` package is used to declare attributes for the class.
@define
class IdealizedActuatorDiskModel(BaseOperationModel):
    """
    Idealized actuator disk model that does not curtail (i.e. has no rated wind speed).
    """

    # Declare attributes for the class using the `field` function from the `attrs` package.
    constant_axial_induction: floris_float_type = field(default=1.0 / 3.0)
    rotor_diameter: floris_float_type = field(default=126.0) # Default to NREL 5MW rotor diameter

    # Declare the required `power`, `thrust`, and `axial_induction` methods.
    def power(self, velocities, air_density, **_):
        """
        Compute the power output of the turbine in Watts.
        """
        rotor_average_velocities = average_velocity(velocities)
        axial_induction = self.axial_induction(velocities)
        power_coefficient = 4 * axial_induction * (1 - axial_induction) ** 2
        return 0.5 * air_density * self._area() * rotor_average_velocities**3 * power_coefficient

    def thrust_coefficient(self, velocities, air_density, **_):
        """
        Compute the thrust force on the turbine in Newtons.
        """
        rotor_average_velocities = average_velocity(velocities)
        axial_induction = self.axial_induction(velocities)
        thrust_coefficient = 4 * axial_induction * (1 - axial_induction)
        return 0.5 * air_density * self._area() * rotor_average_velocities**2 * thrust_coefficient

    def axial_induction(self, velocities, **_):
        """
        Return the (user-provided) axial induction factor of the turbine.
        """
        return self.constant_axial_induction * np.ones(velocities.shape[0:2])

    def _area(self):
        """Compute the rotor swept area of the turbine."""
        return np.pi * (self.rotor_diameter / 2) ** 2


# Create a sweep over wind speed to evaluate the turbine operation model
ws_array = np.arange(0.1, 30.0, 0.2)
wd_array = 270.0 * np.ones_like(ws_array)
turbulence_intensities = 0.06 * np.ones_like(ws_array)

# Instantiate FLORIS with a single wind turbine
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(
    layout_x=[0],
    layout_y=[0],
    wind_speeds=ws_array,
    wind_directions=wd_array,
    turbulence_intensities=turbulence_intensities
)

# First, solve with the default operation model and store the results
fmodel.run()
powers = fmodel.get_turbine_powers()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(ws_array, powers/1e3, color="k", linestyle="--", label="Default operation model")

# Now, create an instance of the user-defined operation model and supply it to FLORIS
fmodel.set_operation_model(IdealizedActuatorDiskModel())
fmodel.run()
powers = fmodel.get_turbine_powers()
ax.plot(ws_array, powers/1e3, label="User-defined operation model (defaults)")

# Change the axial induction factor
fmodel.set_operation_model(IdealizedActuatorDiskModel(constant_axial_induction=0.2))
fmodel.run()
powers = fmodel.get_turbine_powers()
ax.plot(ws_array, powers/1e3, label="User-defined operation model (low axial induction)")

# Reset the axial induction factor; reduce the rotor diameter to 100 m
fmodel.set_operation_model(IdealizedActuatorDiskModel(rotor_diameter=100.0))
fmodel.run()
powers = fmodel.get_turbine_powers()
ax.plot(ws_array, powers/1e3, label="User-defined operation model (smaller rotor)")

# Plot aesthetics
ax.set_xlabel("Wind speed [m/s]")
ax.set_ylabel("Power [kW]")
ax.set_ylim([-2e3, 10e3])
ax.set_xlim([ws_array[0], ws_array[-1]])
ax.grid(True)
ax.legend(loc="upper right")

plt.show()
