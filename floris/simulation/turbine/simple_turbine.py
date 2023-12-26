from scipy.interpolate import interp1d

from floris.simulation.turbine.rotor_effective_velocity import rotor_effective_velocity

class SimpleTurbine():

    @staticmethod
    def power(turbine_model_parameters, velocities):

        # TEMPORARY
        turbine_model_parameters = {"power_thrust_table": turbine_model_parameters}
        
        power_interpolator = interp1d(
            turbine_model_parameters["power_thrust_table"]["wind_speed"],
            turbine_model_parameters["power_thrust_table"]["power"] * 1e3, # Convert to W
            fill_value=0.0,
            bounds_error=False,
        )

        # TODO: covert from "raw" velocities to rotor effective velocities.
        # rotor_effective_velocity(


        # )
        

        power = power_interpolator(velocities)
        
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
