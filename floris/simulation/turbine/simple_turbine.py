from scipy.interpolate import interp1d

class SimpleTurbine():

    @staticmethod
    def power(power_thrust_table, wind_speed_sample):
        
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"] * 1e3, # Convert to W
            fill_value=0.0,
            bounds_error=False,
        )

        power = power_interpolator(wind_speed_sample)
        
        return power

    @staticmethod
    def thrust_coefficient(power_thrust_table, wind_speed_sample):
    
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )
    
        thrust_coefficient = thrust_coefficient_interpolator(wind_speed_sample)
        
        return thrust_coefficient
    
    # TODO: Implement prepare functions
