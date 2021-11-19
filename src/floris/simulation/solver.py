from abc import ABC, abstractmethod

import numpy as np

from floris.simulation import Farm
from floris.simulation import TurbineGrid
from floris.simulation import Ct, axial_induction
from floris.simulation import FlowField
from floris.simulation.wake_velocity.jensen import JensenVelocityDeficit
from floris.simulation.wake_deflection.jimenez import JimenezVelocityDeflection


jimenez_deflection_model = JimenezVelocityDeflection()
jensen_deficit_model = JensenVelocityDeficit()


def sequential_solver(farm: Farm, flow_field: FlowField, grid: TurbineGrid) -> None:

    # <<interface>>
    jimenez_args = jimenez_deflection_model.prepare_function(grid, farm.rotor_diameter[:, :, [0]], farm.farm_controller.yaw_angles)
    jensen_args = jensen_deficit_model.prepare_function(grid, farm.rotor_diameter[:, :, [0]], flow_field)

    # This is u_wake
    velocity_deficit = np.zeros_like(flow_field.u_initial)

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        u = flow_field.u_initial - velocity_deficit
        u[:, :, i+1:, :, :] = 0  # TODO: explain

        thrust_coefficient = Ct(
            velocities=u,
            yaw_angle=farm.farm_controller.yaw_angles,
            fCt=farm.fCt_interp,
            ix_filter=[i],
        )
        deflection_field = jimenez_deflection_model.function(  # n wind speeds, n turbines, grid x, grid y
            i, thrust_coefficient, **jimenez_args
        )

        # jensen_args['u'] = flow_field.u_initial - velocity_deficit
        c = jensen_deficit_model.function(i, deflection_field, **jensen_args)

        turbine_ai = axial_induction(
            velocities=u,
            yaw_angle=farm.farm_controller.yaw_angles,
            fCt=farm.fCt_interp,
            ix_filter=[i],
        )
        turbine_ai = turbine_ai[:, :, :, None, None] * np.ones(
            (
                flow_field.n_wind_directions,
                flow_field.n_wind_speeds,
                grid.n_turbines,
                grid.grid_resolution,
                grid.grid_resolution,
            )
        )

        turb_u_wake = 2 * turbine_ai * c * flow_field.u_initial

        velocity_deficit = np.sqrt(( velocity_deficit ** 2) + (turb_u_wake ** 2))

    flow_field.u = flow_field.u_initial - velocity_deficit
