from abc import ABC, abstractmethod

import numpy as np

from floris.simulation import Farm
from floris.simulation import TurbineGrid
from floris.simulation import Ct, axial_induction
from floris.simulation import FlowField
from floris.simulation.wake_velocity.jensen import JensenVelocityDeficit
from floris.simulation.wake_velocity.gauss import GaussVelocityDeficit
from floris.simulation.wake_deflection.jimenez import JimenezVelocityDeflection
from floris.simulation.wake_deflection.gauss import GaussVelocityDeflection


jensen_deficit_model = JensenVelocityDeficit()
gauss_deficit_model = GaussVelocityDeficit()

jimenez_deflection_model = JimenezVelocityDeflection()
gauss_deflection_model = GaussVelocityDeflection()

# deficit_model = "jensen"
deficit_model = "gauss"

# <<interface>>
if deficit_model == "jensen":
    velocity_deficit_model = jensen_deficit_model
    deflection_model = jimenez_deflection_model
elif deficit_model == "gauss":
    velocity_deficit_model = gauss_deficit_model
    deflection_model = gauss_deflection_model

def sequential_solver(farm: Farm, flow_field: FlowField, grid: TurbineGrid) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    deflection_model_args = deflection_model.prepare_function(grid, farm, flow_field)
    deficit_model_args = velocity_deficit_model.prepare_function(grid, farm, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial)

    turbine_turbulence_intensity = 0.1 * np.ones_like(grid.x)
    ambient_turbulence_intensity = 0.1 * np.ones_like(grid.x)

    reference_rotor_diameter = farm.reference_turbine_diameter * np.ones(
        (
            flow_field.n_wind_directions,
            flow_field.n_wind_speeds,
            grid.n_turbines,
            1,
            1
        )
    )
    
    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        u = flow_field.u_initial - wake_field

        thrust_coefficient = Ct(
            velocities=u,
            yaw_angle=farm.farm_controller.yaw_angles,
            fCt=farm.fCt_interp,
            ix_filter=[i],
        )
        if deficit_model == "jensen":
            deflection_field = deflection_model.function(
                i,
                thrust_coefficient,
                **deflection_model_args
            )
        elif deficit_model == "gauss":
            deflection_field = deflection_model.function(
                i,
                turbine_turbulence_intensity[:, :, i:i+1, :, :],
                thrust_coefficient,
                **deflection_model_args
            )

        turbine_ai = axial_induction(
            velocities=u,
            yaw_angle=farm.farm_controller.yaw_angles,
            fCt=farm.fCt_interp,
            ix_filter=[i],
        )

        if deficit_model == "jensen":
            velocity_deficit = velocity_deficit_model.function(
                i,
                deflection_field,
                turbine_ai,
                **deficit_model_args
            )
        elif deficit_model == "gauss":
            velocity_deficit = velocity_deficit_model.function(
                i,
                deflection_field,
                turbine_turbulence_intensity[:, :, i:i+1, :, :],
                thrust_coefficient,
                **deficit_model_args
            )

        # Sum of squares combination model to incorporate the current turbine's velocity into the main array
        wake_field = np.sqrt( wake_field ** 2 + (velocity_deficit * flow_field.u_initial) ** 2 )

        x_i = np.mean(grid.x[:, :, i:i+1], axis=(3,4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y[:, :, i:i+1], axis=(3,4))
        y_i = y_i[:, :, :, None, None]

        wake_added_turbulence_intensity = crespo_hernandez(
            ambient_turbulence_intensity,
            grid.x,
            x_i,
            reference_rotor_diameter,
            turbine_ai
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        # turb_wake_field = flow_field.u_initial - wake_field
        # area_overlap = calculate_area_overlap(
        #     turb_wake_field, flow_field.u_initial, 5, 5
        # )
        area_overlap = np.sum(velocity_deficit * flow_field.u_initial > 0.05, axis=(3, 4)) / (grid.grid_resolution * grid.grid_resolution)
        area_overlap = area_overlap[:, :, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * reference_rotor_diameter
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * np.array(grid.x > x_i)
            * np.array(np.abs(y_i - grid.y) < 2 * reference_rotor_diameter)
            * np.array(grid.x <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum( np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ) , turbine_turbulence_intensity )

    flow_field.u = flow_field.u_initial - wake_field


def crespo_hernandez(ambient_TI, x_coord_downstream, x_coord_upstream, rotor_diameter, aI):
    ti_initial = 0.1
    ti_constant = 0.5
    ti_ai = 0.8
    ti_downstream = -0.32

    # turbulence intensity calculation based on Crespo et. al.
    ti_calculation = (
        ti_constant
      * aI ** ti_ai
      * ambient_TI ** ti_initial
      * ((x_coord_downstream - x_coord_upstream) / rotor_diameter) ** ti_downstream
    )
    return ti_calculation

def calculate_area_overlap(wake_velocities, freestream_velocities, y_ngrid, z_ngrid):
    """
    compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
    """
    # Count all of the rotor points with a negligible difference from freestream
    # count = np.sum(freestream_velocities - wake_velocities <= 0.05, axis=(3, 4))
    # return (y_ngrid * z_ngrid - count) / (y_ngrid * z_ngrid)
    # return 1 - count / (y_ngrid * z_ngrid)

    # Find the points on the rotor grids with a difference from freestream of greater
    # than some tolerance. These are all the points in the wake. The ratio of
    # these points to the total points is the portion of wake overlap.
    return np.sum(freestream_velocities - wake_velocities > 0.05, axis=(3, 4)) / (y_ngrid * z_ngrid)
    