import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from BaseObject import BaseObject
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
        for coord, turbine in self.turbineMap.items():
            grid = turbine.get_grid()

            # use the z coordinate of the turbine grid points for initialization
            velocities = [self.windSpeed * ((turbine.hubHeight+g[1])/self.characteristicHeight)**self.shear for g in grid]

            turbine.set_velocities(velocities)

    def _initialize_turbines(self):
        for coord, turbine in self.turbineMap.items():
            turbine.initialize()

    def calculate_wake(self):
        # TODO: rotate layout here
        # TODO: sort in ascending order of x coord

        for coord, turbine in self.turbineMap.items():
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
        turbine = self.turbineMap[(0, 0)]
        coords = (0,0)
        x = np.linspace(-2 * turbine.rotorDiameter, coords[0] + 10 * turbine.rotorDiameter, 200)
        y = np.linspace(-250, coords[1] + 250, 200)
        z = np.linspace(0, 2 * turbine.hubHeight, 50)
        return np.meshgrid(x, y, z, indexing='xy')

    def plot_flow_field_plane(self):
        turbine = self.turbineMap[(0,0)]
        x, y, z = self.discretize_domain()
        xmax, ymax, zmax = x.shape[0], y.shape[0], z.shape[0]

        # calculate the velocities on the mesh
        u_free = np.zeros((xmax, ymax, zmax))
        u_free.fill(self.windSpeed)
        u_turb = self.compute_discrete_velocity_field(x, y, z, self.wake)

        self.vizManager.plot_constant_z(x[:, :, 24], y[:, :, 24], u_turb[:, :, 24])
        self.vizManager.add_turbine_marker(turbine.rotorDiameter / 2., (0, 0))
        self.vizManager.show_plot()

    def compute_discrete_velocity_field(self, x, y, z, wake=None):
        turbine = self.turbineMap[(0, 0)]
        xmax, ymax, zmax = x.shape[0], y.shape[0], z.shape[0]        

        velocity_function = wake.get_velocity_function()

        c = velocity_function(x, y, z, turbine.rotorDiameter, 0)
        u = self.windSpeed * (1 - 2 * turbine.aI * c)

        return u
