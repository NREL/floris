from __future__ import annotations

from collections.abc import Iterable

from scipy.interpolate import interp1d

from floris.simulation.turbine.rotor_effective_velocity import (
    air_density_velocity_correction,
    average_velocity,
    rotor_effective_velocity,
)

from floris.type_dec import (
    floris_numeric_dict_converter,
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)

class SimpleTurbine():

    @staticmethod
    def power(
        turbine_power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None
    ):
        # Construct power interpolant
        power_interpolator = interp1d(
            turbine_power_thrust_table["wind_speed"],
            turbine_power_thrust_table["power"] * 1e3, # Convert to W
            fill_value=0.0,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = air_density_velocity_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=turbine_power_thrust_table["ref_air_density"]
        )
        
        # Compute power
        power = power_interpolator(rotor_effective_velocities)
        
        return power

    @staticmethod
    def thrust_coefficient(power_thrust_table, velocities):
    
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )
    
        thrust_coefficient = thrust_coefficient_interpolator(velocities)
        
        return thrust_coefficient
    
    # TODO: Implement prepare functions
