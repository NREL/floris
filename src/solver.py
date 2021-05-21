
from abc import ABC, abstractmethod
from src.turbine import Turbine
import numpy as np
from .farm import Farm
from .flow_field import FlowField
from .grid import TurbineGrid, FlowFieldGrid
from .utilities import Vec3
from .wake_velocity.jensen import JensenVelocityDeficit


def sequential_solver(farm: Farm, flow_field: FlowField) -> None:

    grid = TurbineGrid(farm.coords, flow_field.reference_turbine_diameter, flow_field.reference_wind_height, 5)

    jensen_deficit_model = JensenVelocityDeficit({}, grid, farm, flow_field)

    flow_field.initialize_velocity_field(grid)

    # Calculate the wake deflection field
    deflection = np.array([0.0])

    # total_deficit = np.zeros(np.shape(grid.x))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(1, grid.n_turbines):

        jensen_deficit_model.function(i, farm, flow_field, deflection)
        # print(deficit)
        # total_deficit[i] = total_deficit[i] + deficit

    # flow_field.u[i] = flow_field.u[i-1] * (1 - 2 * turbine_ai * c)
    # print(total_deficit)
        