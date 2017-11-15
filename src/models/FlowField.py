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
        self.turbineMap = turbine_map
        # {
        #   (x,y): {Turbine, TurbineSolution(), Wake()},
        #   (x,y): {Turbine, TurbineSolution(), Wake()},
        #   ...
        # }

        # FlowfieldPropertiesAtTurbine: {
        #     (0, 0): {
        #         Turbine,
        #         ti,
        #         coordinates,
        #         velocity,
        #         get_ct(self.velocity): return turbine.Ct,
        #         get_cp(self.velocity): return turbine.Cp,
        #         get_power,
        #         wake_function
        #     },
        #     (0,10): Turbine,
        #     (0,20): Turbine,
        # }

        self.characteristicHeight = characteristic_height
        self.wake = wake
        if self.valid():
            self.initialize_turbines()
            self.initialize_turbine_velocities()

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
        # TODO: this should only be applied to any turbine seeing freestream

        # initialize the flow field used in the 3D model based on shear using the power log law.
        for coord, turbine in self.turbineMap.items():
            grid = turbine.get_grid()

            # use the z coordinate of the turbine grid points for initialization
            velocities = [self.windSpeed * ((turbine.hubHeight+g[1])/self.characteristicHeight)**self.shear for g in grid]

            turbine.set_velocities(velocities)

    def initialize_turbines(self):
        for coord, turbine in self.turbineMap.items():
            turbine.initialize_velocities()

    def calculate_wake(self):
        # TODO: rotate layout here
        # TODO: sort in ascending order of x coord

        for coord, turbine in self.turbineMap.items():
            # TODO: store current turbine TI
            # local_ti = 0
            # local_velocity = 0
            previous_turbines_x = 0

            # calculate wake at this turbine
            print(coord, self.wake.calculate(10, turbine.rotorDiameter, turbine.Ct, coord[0]))

            # TODO: calculate wake at all downstream turbines


            # TODO: if last turbine, break

        # for a turbine that doesnt have a TI, find all turbines that impact this turbine's swept area
        # generate a new TI ...


    # def update_flowfield():

    def get_properties_at_turbine(tuple_of_coords):
        #probe the FlowField
        FlowfieldPropertiesAtTurbine[tuple_of_coords].wake_function()
