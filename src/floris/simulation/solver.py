from abc import ABC, abstractmethod

import numpy as np

from floris.simulation import Farm
from floris.simulation import TurbineGrid
from floris.simulation import Ct, axial_induction
from floris.simulation import FlowField
from floris.simulation.wake_velocity.jensen import JensenVelocityDeficit
from floris.simulation.wake_velocity.gaussianModels.gauss import GaussVelocityDeficit
from floris.simulation.wake_deflection.jimenez import JimenezVelocityDeflection


jimenez_deflection_model = JimenezVelocityDeflection()
jensen_deficit_model = JensenVelocityDeficit()
gauss_deficit_model = GaussVelocityDeficit()

deficit_model = "jensen"
# deficit_model = "gauss"

# <<interface>>
if deficit_model == "jensen":
    velocity_deficit_model = jensen_deficit_model
elif deficit_model == "gauss":
    velocity_deficit_model = gauss_deficit_model
deflection_model = jimenez_deflection_model

def sequential_solver(farm: Farm, flow_field: FlowField, grid: TurbineGrid) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    jimenez_args = deflection_model.prepare_function(grid, farm, flow_field)
    deficit_model_args = velocity_deficit_model.prepare_function(grid, farm, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial)

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        u = flow_field.u_initial - wake_field

        thrust_coefficient = Ct(
            velocities=u,
            yaw_angle=farm.farm_controller.yaw_angles,
            fCt=farm.fCt_interp,
            ix_filter=[i],
        )
        deflection_field = deflection_model.function(i, thrust_coefficient, **jimenez_args)

        turbine_ai = axial_induction(
            velocities=u,
            yaw_angle=farm.farm_controller.yaw_angles,
            fCt=farm.fCt_interp,
            ix_filter=[i],
        )
        turbine_ai = turbine_ai[:, :, :, None, None]

        if deficit_model == "jensen":
            velocity_deficit = velocity_deficit_model.function(i, deflection_field, turbine_ai, **deficit_model_args)
        elif deficit_model == "gauss":
            velocity_deficit = velocity_deficit_model.function(i, deflection_field, thrust_coefficient, **deficit_model_args)

        # Sum of squares combination model to incorporate the current turbine's velocity into the main array
        wake_field = np.sqrt( wake_field ** 2 + (velocity_deficit * flow_field.u_initial) ** 2 )

    flow_field.u = flow_field.u_initial - wake_field
