import copy
from abc import abstractmethod
from typing import Callable

import numpy as np
from attrs import (
    define,
    field,
    fields,
)

from floris.core import (
    axial_induction,
    BaseLibrary,
    Farm,
    FlowField,
    FlowFieldPlanarGrid,
    PointsGrid,
    thrust_coefficient,
    TurbineGrid,
)


@define
class BaseWakeModel(BaseLibrary): # Inherit instead from BaseLibrary

    # Storage
    x_i: np.ndarray = field(init=False, default=None)
    y_i: np.ndarray = field(init=False, default=None)
    z_i: np.ndarray = field(init=False, default=None)

    yaw_angle_i: np.ndarray = field(init=False, default=None)
    hub_height_i: np.ndarray = field(init=False, default=None)
    rotor_diameter_i: np.ndarray = field(init=False, default=None)
    TSR_i: np.ndarray = field(init=False, default=None)

    # Combination model
    combination_function: Callable = field(init=False, default=None)

    def set_turbine_i(self, grid, farm, i):

        # Get the current turbine quantities
        self.x_i = np.mean(grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        self.y_i = np.mean(grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        self.z_i = np.mean(grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        self.yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
        self.hub_height_i = farm.hub_heights_sorted[:, i:i+1, None, None]
        self.rotor_diameter_i = farm.rotor_diameters_sorted[:, i:i+1, None, None]
        self.TSR_i = farm.TSRs_sorted[:, i:i+1, None, None]

    def assign_combination_function(self, combination_function):
        self.combination_function = combination_function

    @staticmethod
    def turbine_thrust_coefficient(grid, farm, flow_field, i):
        # TODO: consider just calling the appropriate method on the operation model,
        # instead?
        ct_i = thrust_coefficient(
            turbines=farm.turbines,
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions
        )
        # Since we are filtering for the i'th turbine in the thrust coefficient function,
        # get the first index here (0:1)
        return ct_i[:, 0:1, None, None]

    @abstractmethod
    def turbine_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: TurbineGrid,
    ) -> None:
        raise NotImplementedError(
            "The turbine_solve method has not yet been implemented for "+self.__class__.__name__
        )

    @abstractmethod
    def point_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: FlowFieldPlanarGrid | PointsGrid,
    ):
        raise NotImplementedError(
            "points_solve is not implemented for "+self.__class__.__name__
        )

    @staticmethod
    def turbine_axial_induction(grid, farm, flow_field, i):
        # TODO: consider just calling the appropriate method on the operation model,
        # instead?
        axial_induction_i = axial_induction(
            turbines=farm.turbines,
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions
        )

        return axial_induction_i[:, 0:1, None, None]

    @staticmethod
    def generate_turbine_grid_objects(
        farm: Farm,
        flow_field: FlowField,
    ):
        """Generate turbine grid objects from points grid objects.
           Intermediate step of point_solve.
        """
        turbine_grid_farm = copy.deepcopy(farm)
        turbine_grid_flow_field = copy.deepcopy(flow_field)

        turbine_grid = TurbineGrid(
            turbine_coordinates=turbine_grid_farm.coordinates,
            turbine_diameters=turbine_grid_farm.rotor_diameters,
            wind_directions=turbine_grid_flow_field.wind_directions,
            grid_resolution=3,
        )
        turbine_grid_farm.set_sorted_indices(turbine_grid.sorted_coord_indices)
        turbine_grid_farm.construct_turbine_type_map()
        turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
        turbine_grid_farm.initialize()

        return turbine_grid_farm, turbine_grid_flow_field, turbine_grid
