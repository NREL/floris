from BaseObject import BaseObject


class FlowField(BaseObject):
    """
        Describe FF here
    """

    def __init__(self,
                 wake_combination=None,
                 wind_speed=None,
                 shear=None,
                 turbine_map=None,
                 characteristic_height=None,
                 wake=None):
        super().__init__()
        self.wakeCombination = wake_combination
        self.windSpeed = wind_speed
        self.shear = shear
        self.turbineMap = turbine_map  # {(x,y): {Turbine, U}, (x,y): Turbine, ... }
        self.characteristicHeight = characteristic_height
        self.wake = wake
        if self.valid():
            self.initialize_turbine_velocities()
            self.initialize_turbines()

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            return False
        if self.characteristicHeight <= 0:
            valid = False
        return valid

    def initialize_turbine_velocities(self):
        # TODO: why do we actually need to initialize?

        # initialize the flow field used in the 3D model based on shear using the power log law.
        for coord, turbine in self.turbineMap.items():
            grid = turbine.get_grid()

            # use the z coordinate of the turbine grid points for initialization
            velocities = [self.windSpeed * ((turbine.hubHeight+g[1])/self.characteristicHeight)**self.shear for g in grid]

            turbine.set_velocities(velocities)

    def initialize_turbines(self):
        for coord, turbine in self.turbineMap.items():
            turbine.initialize()
