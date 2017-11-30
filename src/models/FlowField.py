import os
import sys
from BaseObject import BaseObject
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.io.VisualizationManager import VisualizationManager


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
        self.vizManager = VisualizationManager()
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
            self._initialize_turbine_velocities()
            self._initialize_turbines()

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

    def _initialize_turbine_velocities(self):
        # TODO: this should only be applied to any turbine seeing freestream

        # initialize the flow field used in the 3D model based on shear using the power log law.
        for _, turbine in self.turbineMap.items():
            grid = turbine.get_grid()

            # use the z coordinate of the turbine grid points for initialization
            velocities = [self.windSpeed * ((turbine.hubHeight+g[1])/self.characteristicHeight)**self.shear for g in grid]

            turbine.set_velocities(velocities)

    def _initialize_turbines(self):
        for _, turbine in self.turbineMap.items():
            turbine.initialize()

    def calculate_wake(self):
        # TODO: rotate layout here
        # TODO: sort in ascending order of x coord

        for _, turbine in self.turbineMap.items():
            # TODO: store current turbine TI
            # local_ti = 0
            # local_velocity = 0
            previous_turbines_x = 0

            # calculate wake at this turbine
            # def _jensen(self, streamwise_location, horizontal_location, vertical_location, turbine_diameter, turbine_x):
            # print(coord, self.wake.calculate(10, 0, turbine.hubHeight, turbine.rotorDiameter, coord[0]))

            # TODO: calculate wake at all downstream turbines

            # TODO: if last turbine, break

        # for a turbine that doesnt have a TI, find all turbines that impact this turbine's swept area
        # generate a new TI ...


    # def update_flowfield():

    def get_properties_at_turbine(tuple_of_coords):
        #probe the FlowField
        FlowfieldPropertiesAtTurbine[tuple_of_coords].wake_function()


    # visualization

    def discretize_domain(self):
        coords = [coord for coord, _ in self.turbineMap.items()]
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        turbines = [turbine for _, turbine in self.turbineMap.items()]
        maxDiameter = max([turbine.rotorDiameter for turbine in turbines])
        hubHeight = turbines[0].hubHeight
        
        x = np.linspace(xmin - 2 * maxDiameter, xmax + 10 * maxDiameter, 200)
        y = np.linspace(ymin - 2 * maxDiameter, ymax + 2 * maxDiameter, 200)
        z = np.linspace(0, 2 * hubHeight, 50)
        return np.meshgrid(x, y, z, indexing='xy')

    def plot_flow_field_plane(self):
        x, y, z = self.discretize_domain()
        xmax, ymax, zmax = x.shape[0], y.shape[1], z.shape[2]

        velocity_function = self.wake.get_velocity_function()

        # calculate the velocities on the mesh
        u_field = np.full((xmax, ymax, zmax), self.windSpeed)

        for coord, turbine in self.turbineMap.items():
            u_wake = self.compute_turbine_velocity_deficit(x, y, z, velocity_function, turbine, coord)
            u_field = self.wakeCombination.combine(None, None, u_field, u_wake)

        self.vizManager.plot_constant_z(x[:, :, 24], y[:, :, 24], u_field[:, :, 24])
        for coord, turbine in self.turbineMap.items():
            self.vizManager.add_turbine_marker(turbine.rotorRadius, coord)
        self.vizManager.show_plot()

    def compute_turbine_velocity_deficit(self, x, y, z, velocity_function, turbine, coord):
        """
            computes the discrete velocity field x, y, z for turbine using velocity_function
        """
        return self.windSpeed * 2 * turbine.aI * \
            velocity_function(x, y, z, turbine.rotorRadius, coord)
