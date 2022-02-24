# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import copy
import numpy as np
import time
import sys

from floris.simulation import Farm
from floris.simulation import Turbine
from floris.simulation import TurbineGrid, FlowFieldGrid
from floris.simulation import Ct, axial_induction
from floris.simulation import FlowField
from floris.simulation.turbine import average_velocity
from floris.simulation.wake import WakeModelManager
from floris.simulation.wake_deflection.gauss import (
    calculate_transverse_velocity,
    wake_added_yaw,
    yaw_added_turbulence_mixing
)


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


# @profile
def sequential_solver(farm: Farm, flow_field: FlowField, turbine: Turbine, grid: TurbineGrid, model_manager: WakeModelManager) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field, turbine)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field, turbine)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial)
    v_wake = np.zeros_like(flow_field.v_initial)
    w_wake = np.zeros_like(flow_field.w_initial)

    turbine_turbulence_intensity = flow_field.turbulence_intensity * np.ones((flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1))
    ambient_turbulence_intensity = flow_field.turbulence_intensity

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y[:, :, i:i+1], axis=(3, 4))        
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = flow_field.u[:, :, i:i+1]
        v_i = flow_field.v[:, :, i:i+1]

        ct_i = Ct(
            velocities=flow_field.u,
            yaw_angle=farm.yaw_angles,
            fCt=turbine.fCt_interp,
            ix_filter=[i],
        )
        ct_i = ct_i[:, :, 0:1, None, None]  # Since we are filtering for the i'th turbine in the Ct function, get the first index here (0:1)
        axial_induction_i = axial_induction(
            velocities=flow_field.u,
            yaw_angle=farm.yaw_angles,
            fCt=turbine.fCt_interp,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, 0:1, None, None]    # Since we are filtering for the i'th turbine in the axial induction function, get the first index here (0:1)
        turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
        yaw_angle_i = farm.yaw_angles[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial,
                grid.y[:, :, i:i+1] - y_i,
                grid.z[:, :, i:i+1],
                turbine.rotor_diameter,
                turbine.hub_height,
                ct_i,
                turbine.TSR,
                axial_induction_i
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            ct_i,
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial,
                grid.x - x_i,
                grid.y - y_i,
                grid.z,
                turbine.rotor_diameter,
                turbine.hub_height,
                yaw_angle_i,
                ct_i,
                turbine.TSR,
                axial_induction_i
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w[:, :, i:i+1],
                v_wake[:, :, i:i+1],
                w_wake[:, :, i:i+1],
            )
            gch_gain = 2
            turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field,
            yaw_angle_i,
            turbulence_intensity_i,
            ct_i,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensity,
            grid.x,
            x_i,
            turbine.rotor_diameter,
            axial_induction_i
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = np.sum(velocity_deficit * flow_field.u_initial > 0.05, axis=(3, 4)) / (grid.grid_resolution * grid.grid_resolution)
        area_overlap = area_overlap[:, :, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * turbine.rotor_diameter
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * np.array(grid.x > x_i)
            * np.array(np.abs(y_i - grid.y) < 2 * turbine.rotor_diameter)
            * np.array(grid.x <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum( np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ) , turbine_turbulence_intensity )

        flow_field.u = flow_field.u_initial - wake_field
        flow_field.v += v_wake
        flow_field.w += w_wake

    flow_field.turbulence_intensity_field = np.mean(turbine_turbulence_intensity, axis=(3,4))
    flow_field.turbulence_intensity_field = flow_field.turbulence_intensity_field[:,:,:,None,None]


def full_flow_sequential_solver(farm: Farm, flow_field: FlowField, turbine: Turbine, flow_field_grid: FlowFieldGrid, model_manager: WakeModelManager) -> None:

    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)
    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        reference_turbine_diameter=turbine.rotor_diameter,
        wind_directions=turbine_grid_flow_field.wind_directions,
        wind_speeds=turbine_grid_flow_field.wind_speeds,
        grid_resolution=5,
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    sequential_solver(turbine_grid_farm, turbine_grid_flow_field, turbine, turbine_grid, model_manager)

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(flow_field_grid, flow_field, turbine)
    deficit_model_args = model_manager.velocity_model.prepare_function(flow_field_grid, flow_field, turbine)

    wake_field = np.zeros_like(flow_field.u_initial)
    v_wake = np.zeros_like(flow_field.v_initial)
    w_wake = np.zeros_like(flow_field.w_initial)

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(turbine_grid.x[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(turbine_grid.y[:, :, i:i+1], axis=(3, 4))        
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(turbine_grid.z[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = turbine_grid_flow_field.u[:, :, i:i+1]
        v_i = turbine_grid_flow_field.v[:, :, i:i+1]

        ct_i = Ct(
            velocities=turbine_grid_flow_field.u,
            yaw_angle=turbine_grid_farm.yaw_angles,
            fCt=turbine.fCt_interp,
            ix_filter=[i],
        )
        ct_i = ct_i[:, :, 0:1, None, None]  # Since we are filtering for the i'th turbine in the Ct function, get the first index here (0:1)
        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u,
            yaw_angle=turbine_grid_farm.yaw_angles,
            fCt=turbine.fCt_interp,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, 0:1, None, None]    # Since we are filtering for the i'th turbine in the axial induction function, get the first index here (0:1)
        turbulence_intensity_i = turbine_grid_flow_field.turbulence_intensity_field[:, :, i:i+1]
        yaw_angle_i = turbine_grid_farm.yaw_angles[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                turbine_grid_flow_field.u_initial,
                turbine_grid.y[:, :, i:i+1] - y_i,
                turbine_grid.z[:, :, i:i+1],
                turbine.rotor_diameter,
                turbine.hub_height,
                ct_i,
                turbine.TSR,
                axial_induction_i
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            ct_i,
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial,
                flow_field_grid.x - x_i,
                flow_field_grid.y - y_i,
                flow_field_grid.z,
                turbine.rotor_diameter,
                turbine.hub_height,
                yaw_angle_i,
                ct_i,
                turbine.TSR,
                axial_induction_i
            )

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field,
            yaw_angle_i,
            turbulence_intensity_i,
            ct_i,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial
        )

        flow_field.u = flow_field.u_initial - wake_field
        flow_field.v += v_wake
        flow_field.w += w_wake


def cc_solver(farm: Farm, flow_field: FlowField, turbine: Turbine, grid: TurbineGrid, model_manager: WakeModelManager) -> None:

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field, turbine)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field, turbine)

    # This is u_wake
    v_wake = np.zeros_like(flow_field.v_initial)
    w_wake = np.zeros_like(flow_field.w_initial)
    turb_u_wake = np.zeros_like(flow_field.u_initial)
    turb_inflow_field = copy.deepcopy(flow_field.u_initial)

    turbine_turbulence_intensity = flow_field.turbulence_intensity * np.ones((flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1))
    ambient_turbulence_intensity = flow_field.turbulence_intensity

    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial)
    Ctmp = np.zeros((shape))
    # Ctmp = np.zeros((len(x_coord), len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))
    
    sigma_i = np.zeros((shape))
    # sigma_i = np.zeros((len(x_coord), len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y[:, :, i:i+1], axis=(3, 4))        
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        mask2 = np.array(grid.x < x_i + 0.01) * np.array(grid.x > x_i - 0.01) * np.array(grid.y < y_i + 0.51*126.0) * np.array(grid.y > y_i - 0.51*126.0)
        # mask2 = np.logical_and(np.logical_and(np.logical_and(grid.x < x_i + 0.01, grid.x > x_i - 0.01), grid.y < y_i + 0.51*126.0), grid.y > y_i - 0.51*126.0)
        turb_inflow_field = turb_inflow_field * ~mask2 + (flow_field.u_initial - turb_u_wake) * mask2

        turb_avg_vels = average_velocity(turb_inflow_field)
        turb_Cts = Ct(
            turb_avg_vels,
            farm.yaw_angles,
            turbine.fCt_interp,
        )
        turb_Cts = turb_Cts[:, :, :, None, None]     
        turb_aIs = axial_induction(
            turb_avg_vels,
            farm.yaw_angles,
            turbine.fCt_interp,
            ix_filter=[i],
        )
        turb_aIs = turb_aIs[:, :, :, None, None]

        u_i = turb_inflow_field[:, :, i:i+1]
        v_i = flow_field.v[:, :, i:i+1]

        axial_induction_i = axial_induction(
            velocities=flow_field.u,
            yaw_angle=farm.yaw_angles,
            fCt=turbine.fCt_interp,
            ix_filter=[i],
        )

        axial_induction_i = axial_induction_i[:, :, :, None, None]

        turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
        yaw_angle_i = farm.yaw_angles[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial,
                grid.y[:, :, i:i+1] - y_i,
                grid.z[:, :, i:i+1],
                turbine.rotor_diameter,
                turbine.hub_height,
                turb_Cts[:, :, i:i+1],
                turbine.TSR,
                axial_induction_i,
                scale=2.0,
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, :, i:i+1],
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial,
                grid.x - x_i,
                grid.y - y_i,
                grid.z,
                turbine.rotor_diameter,
                turbine.hub_height,
                yaw_angle_i,
                turb_Cts[:, :, i:i+1],
                turbine.TSR,
                axial_induction_i,
                scale=2.0
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w[:, :, i:i+1],
                v_wake[:, :, i:i+1],
                w_wake[:, :, i:i+1],
            )
            gch_gain = 1.0
            turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        turb_u_wake, Ctmp = model_manager.velocity_model.function(
            i,
            x_i,
            y_i,
            z_i,
            u_i,
            deflection_field,
            yaw_angle_i,
            turbine_turbulence_intensity,
            turb_Cts,
            turb_u_wake,
            Ctmp,
            **deficit_model_args
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensity,
            grid.x,
            x_i,
            turbine.rotor_diameter,
            turb_aIs
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = 1 - np.sum(turb_u_wake <= 0.05, axis=(3, 4)) / (grid.grid_resolution * grid.grid_resolution)
        area_overlap = area_overlap[:, :, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * turbine.rotor_diameter
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * np.array(grid.x > x_i)
            * np.array(np.abs(y_i - grid.y) < 2 * turbine.rotor_diameter)
            * np.array(grid.x <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum( np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ) , turbine_turbulence_intensity )

        flow_field.v += v_wake
        flow_field.w += w_wake
    flow_field.u = turb_inflow_field

    flow_field.turbulence_intensity_field = np.mean(turbine_turbulence_intensity, axis=(3,4))
    flow_field.turbulence_intensity_field = flow_field.turbulence_intensity_field[:,:,:,None,None]


def full_flow_cc_solver(farm: Farm, flow_field: FlowField, turbine: Turbine, flow_field_grid: FlowFieldGrid, model_manager: WakeModelManager) -> None:

    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)
    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        reference_turbine_diameter=turbine.rotor_diameter,
        wind_directions=turbine_grid_flow_field.wind_directions,
        wind_speeds=turbine_grid_flow_field.wind_speeds,
        grid_resolution=3,
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    cc_solver(turbine_grid_farm, turbine_grid_flow_field, turbine, turbine_grid, model_manager)

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(flow_field_grid, flow_field, turbine)
    deficit_model_args = model_manager.velocity_model.prepare_function(flow_field_grid, flow_field, turbine)

    v_wake = np.zeros_like(flow_field.v_initial)
    w_wake = np.zeros_like(flow_field.w_initial)
    turb_u_wake = np.zeros_like(flow_field.u_initial)

    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial)
    Ctmp = np.zeros((shape))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(turbine_grid.x[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(turbine_grid.y[:, :, i:i+1], axis=(3, 4))        
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(turbine_grid.z[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = turbine_grid_flow_field.u[:, :, i:i+1]
        v_i = turbine_grid_flow_field.v[:, :, i:i+1]

        turb_avg_vels = average_velocity(turbine_grid_flow_field.u)
        turb_Cts = Ct(
            velocities=turb_avg_vels,
            yaw_angle=turbine_grid_farm.yaw_angles,
            fCt=turbine.fCt_interp,
        )
        turb_Cts = turb_Cts[:, :, :, None, None]

        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u,
            yaw_angle=turbine_grid_farm.yaw_angles,
            fCt=turbine.fCt_interp,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, :, None, None]

        turbulence_intensity_i = turbine_grid_flow_field.turbulence_intensity_field[:, :, i:i+1]
        yaw_angle_i = turbine_grid_farm.yaw_angles[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                turbine_grid_flow_field.u_initial,
                turbine_grid.y[:, :, i:i+1] - y_i,
                turbine_grid.z[:, :, i:i+1],
                turbine.rotor_diameter,
                turbine.hub_height,
                turb_Cts[:, :, i:i+1],
                turbine.TSR,
                axial_induction_i,
                scale=2.0
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, :, i:i+1],
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial,
                flow_field_grid.x - x_i,
                flow_field_grid.y - y_i,
                flow_field_grid.z,
                turbine.rotor_diameter,
                turbine.hub_height,
                yaw_angle_i,
                turb_Cts[:, :, i:i+1],
                turbine.TSR,
                axial_induction_i,
                scale=2.0
            )

        # NOTE: exponential
        turb_u_wake, Ctmp = model_manager.velocity_model.function(
            i,
            x_i,
            y_i,
            z_i,
            u_i,
            deflection_field,
            yaw_angle_i,
            turbine_grid_flow_field.turbulence_intensity_field,
            turb_Cts,
            turb_u_wake,
            Ctmp,
            **deficit_model_args
        )

        flow_field.v += v_wake
        flow_field.w += w_wake
    flow_field.u = flow_field.u_initial - turb_u_wake
