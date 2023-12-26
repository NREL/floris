from scipy.interpolate import interp1d

class SimpleTurbine():

    @staticmethod
    def power(power_thrust_table, wind_speeds_sample):
        
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"] * 1e3, # Convert to W
            fill_value=0.0,
            bounds_error=False,
        )

        power = power_interpolator(wind_speeds_sample)
        
        return power

    @staticmethod
    def thrust_coefficient():
        Ct = 0.1
        return Ct
    
    # TODO: Implement prepare functions
