# Copyright 2021 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from __future__ import annotations

import copy
from typing import Any, Tuple
from pathlib import Path
from itertools import repeat, product
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from numpy.lib.arraysetops import unique

from floris.utilities import Vec3
from floris.type_dec import NDArrayFloat
from floris.simulation import Farm, Floris, FlowField, WakeModelManager, farm, floris, flow_field
from floris.logging_manager import LoggerBase
from floris.tools.cut_plane import get_plane_from_flow_data
# from floris.tools.flow_data import FlowData
from floris.simulation.turbine import Ct, power, axial_induction, average_velocity
from floris.tools.interface_utilities import get_params, set_params, show_params
from floris.tools.cut_plane import CutPlane, change_resolution, get_plane_from_flow_data

# from .visualization import visualize_cut_plane
# from .layout_functions import visualize_layout, build_turbine_loc


class FlorisInterface(LoggerBase):
    """
    FlorisInterface provides a high-level user interface to many of the
    underlying methods within the FLORIS framework. It is meant to act as a
    single entry-point for the majority of users, simplifying the calls to
    methods on objects within FLORIS.

    Args:
        configuration (:py:obj:`dict`): The Floris configuration dictarionary, JSON file,
            or YAML file. The configuration should have the following inputs specified.
                - **flow_field**: See `floris.simulation.flow_field.FlowField` for more details.
                - **farm**: See `floris.simulation.farm.Farm` for more details.
                - **turbine**: See `floris.simulation.turbine.Turbine` for more details.
                - **wake**: See `floris.simulation.wake.WakeManager` for more details.
                - **logging**: See `floris.simulation.floris.Floris` for more details.
    """

    def __init__(self, configuration: dict | str | Path, het_map=None):
        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            self.floris = Floris.from_file(self.configuration)

        elif isinstance(self.configuration, dict):
            self.floris = Floris.from_dict(self.configuration)

        else:
            raise TypeError("The Floris `configuration` must of type 'dict', 'str', or 'Path'.")

        # Store the heterogeneous map for use after reinitailization
        self.het_map = het_map
        # Assign the heterogeneous map to the flow field
        # Needed for a direct call to fi.calculate_wake without fi.reinitialize
        self.floris.flow_field.het_map = het_map

    def calculate_wake(
        self,
        yaw_angles: NDArrayFloat | list[float] | None = None,
        # no_wake: bool = False,
        # points: NDArrayFloat | list[float] | None = None,
        # track_n_upstream_wakes: bool = False,
    ) -> None:
        """
        Wrapper to the :py:meth:`~.Farm.set_yaw_angles` and
        :py:meth:`~.FlowField.calculate_wake` methods.

        Args:
            yaw_angles (NDArrayFloat | list[float] | None, optional): Turbine yaw angles.
                Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            points: (NDArrayFloat | list[float] | None, optional): The x, y, and z
                coordinates at which the flow field velocity is to be recorded. Defaults
                to None.
            track_n_upstream_wakes (bool, optional): When *True*, will keep track of the
                number of upstream wakes a turbine is experiencing. Defaults to *False*.
        """
        # self.floris.flow_field.calculate_wake(
        #     no_wake=no_wake,
        #     points=points,
        #     track_n_upstream_wakes=track_n_upstream_wakes,
        # )

        # TODO decide where to handle this sign issue
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        self.floris.steady_state_atmospheric_condition()

    def reinitialize(
        self,
        wind_speeds: list[float] | NDArrayFloat | None = None,
        wind_directions: list[float] | NDArrayFloat | None = None,
        # wind_layout: list[float] | NDArrayFloat | None = None,
        wind_shear: float | None = None,
        wind_veer: float | None = None,
        reference_wind_height: float | None = None,
        turbulence_intensity: float | None = None,
        # turbulence_kinetic_energy=None,
        air_density: float | None = None,
        # wake: WakeModelManager = None,
        layout: Tuple[list[float], list[float]] | Tuple[NDArrayFloat, NDArrayFloat] | None = None,
        turbine_type: list | None = None,
        # turbine_id: list[str] | None = None,
        # wtg_id: list[str] | None = None,
        # with_resolution: float | None = None,
        solver_settings: dict | None = None
    ):
        # Export the floris object recursively as a dictionary
        floris_dict = self.floris.as_dict()
        flow_field_dict = floris_dict["flow_field"]
        farm_dict = floris_dict["farm"]

        # Make the given changes

        ## FlowField
        if wind_speeds is not None:
            flow_field_dict["wind_speeds"] = wind_speeds
        if wind_directions is not None:
            flow_field_dict["wind_directions"] = wind_directions
        if wind_shear is not None:
            flow_field_dict["wind_shear"] = wind_shear
        if wind_veer is not None:
            flow_field_dict["wind_veer"] = wind_veer
        if reference_wind_height is not None:
            flow_field_dict["reference_wind_height"] = reference_wind_height
        if turbulence_intensity is not None:
            flow_field_dict["turbulence_intensity"] = turbulence_intensity
        if air_density is not None:
            flow_field_dict["air_density"] = air_density

        ## Farm
        if layout is not None:
            farm_dict["layout_x"] = layout[0]
            farm_dict["layout_y"] = layout[1]
        if turbine_type is not None:
            farm_dict["turbine_type"] = turbine_type

        ## Wake
        # if wake is not None:
        #     self.floris.wake = wake
        # if turbulence_intensity is not None:
        #     pass  # TODO: this should be in the code, but maybe got skipped?
        # if turbulence_kinetic_energy is not None:
        #     pass  # TODO: not needed until GCH
        if solver_settings is not None:
            floris_dict["solver"] = solver_settings

        floris_dict["flow_field"] = flow_field_dict
        floris_dict["farm"] = farm_dict

        # Create a new instance of floris and attach to self
        self.floris = Floris.from_dict(floris_dict)
        # Re-assign the hetergeneous inflow map to flow field
        self.floris.flow_field.het_map = self.het_map

    def get_plane_of_points(
        self,
        normal_vector="z",
        planar_coordinate=None,
    ):
        """
        Calculates velocity values through the
        :py:meth:`~.FlowField.calculate_wake` method at points in plane
        specified by inputs.

        Args:
            normal_vector (string, optional): Vector normal to plane.
                Defaults to z.
            planar_coordinate (float, optional): Value of normal vector to slice through. Defaults to None.


        Returns:
            :py:class:`pandas.DataFrame`: containing values of x1, x2, u, v, w
        """
        # Get results vectors
        x_flat = self.floris.grid.x[0, 0].flatten()
        y_flat = self.floris.grid.y[0, 0].flatten()
        z_flat = self.floris.grid.z[0, 0].flatten()
        u_flat = self.floris.flow_field.u[0, 0].flatten()
        v_flat = self.floris.flow_field.v[0, 0].flatten()
        w_flat = self.floris.flow_field.w[0, 0].flatten()

        # Create a df of these
        if normal_vector == "z":
            df = pd.DataFrame(
                {
                    "x1": x_flat,
                    "x2": y_flat,
                    "x3": z_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )
        if normal_vector == "x":
            df = pd.DataFrame(
                {
                    "x1": y_flat,
                    "x2": z_flat,
                    "x3": x_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )
        if normal_vector == "y":
            df = pd.DataFrame(
                {
                    "x1": x_flat,
                    "x2": z_flat,
                    "x3": y_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )

        # Subset to plane
        # TODO: Seems sloppy as need more than one plane in the z-direction for GCH
        if planar_coordinate is not None:
            df = df[np.isclose(df.x3, planar_coordinate)] # , atol=0.1, rtol=0.0)]

        # Drop duplicates
        # TODO is this still needed now that we setup a grid for just this plane?
        df = df.drop_duplicates()

        # Sort values of df to make sure plotting is acceptable
        df = df.sort_values(["x2", "x1"]).reset_index(drop=True)

        return df

    def calculate_horizontal_plane(
        self,
        height,
        x_resolution=200,
        y_resolution=200,
        x_bounds=None,
        y_bounds=None,
        wd=None,
        ws=None,
        yaw_angles=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        #TODO update docstring
        if wd is None:
            wd = self.floris.flow_field.wind_directions
        if ws is None:
            ws = self.floris.flow_field.wind_speeds
        self.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Store the current state for reinitialization
        floris_dict = self.floris.as_dict()

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "z",
            "planar_coordinate": height,
            "flow_field_grid_points": [x_resolution, y_resolution],
            "flow_field_bounds": [x_bounds, y_bounds],
        }
        self.reinitialize(
            wind_directions=wd, wind_speeds=ws, solver_settings=solver_settings
        )

        # TODO this has to be done here as it seems to be lost with reinitialize
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        # Calculate wake
        self.floris.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary? It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = self.get_plane_of_points(
            normal_vector="z",
            planar_coordinate=height,
        )

        # Compute the cutplane
        horizontal_plane = CutPlane(df, self.floris.grid.grid_resolution[0], self.floris.grid.grid_resolution[1])

        # Reset the fi object back to the turbine grid configuration
        self.floris = Floris.from_dict(floris_dict)
        self.floris.flow_field.het_map = self.het_map

        # Run the simulation again for futher postprocessing (i.e. now we can get farm power)
        self.calculate_wake()

        return horizontal_plane

    def calculate_cross_plane(
        self,
        downstream_dist,
        y_resolution=200,
        z_resolution=200,
        y_bounds=None,
        z_bounds=None,
        wd=None,
        ws=None,
        yaw_angles=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        # TODO update docstring
        if wd is None:
            wd = self.floris.flow_field.wind_directions
        if ws is None:
            ws = self.floris.flow_field.wind_speeds

        self.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Store the current state for reinitialization
        floris_dict = self.floris.as_dict()

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "x",
            "planar_coordinate": downstream_dist,
            "flow_field_grid_points": [y_resolution, z_resolution],
            "flow_field_bounds": [y_bounds, z_bounds],
        }
        self.reinitialize(
            wind_directions=wd, wind_speeds=ws, solver_settings=solver_settings
        )

        # TODO this has to be done here as it seems to be lost with reinitialize
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        # Calculate wake
        self.floris.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary? It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = self.get_plane_of_points(
            normal_vector="x",
            planar_coordinate=downstream_dist,
        )

        # Compute the cutplane
        cross_plane = CutPlane(df, y_resolution, z_resolution)

        # Reset the fi object back to the turbine grid configuration
        self.floris = Floris.from_dict(floris_dict)
        self.floris.flow_field.het_map = self.het_map

        # Run the simulation again for futher postprocessing (i.e. now we can get farm power)
        self.calculate_wake()

        return cross_plane

    def calculate_y_plane(
        self,
        crossstream_dist,
        x_resolution=200,
        z_resolution=200,
        x_bounds=None,
        z_bounds=None,
        wd=None,
        ws=None,
        yaw_angles=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        #TODO update docstring
        if wd is None:
            wd = self.floris.flow_field.wind_directions
        if ws is None:
            ws = self.floris.flow_field.wind_speeds
        self.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Store the current state for reinitialization
        floris_dict = self.floris.as_dict()

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "y",
            "planar_coordinate": crossstream_dist,
            "flow_field_grid_points": [x_resolution, z_resolution],
            "flow_field_bounds": [x_bounds, z_bounds],
        }
        self.reinitialize(
            wind_directions=wd, wind_speeds=ws, solver_settings=solver_settings
        )

        # TODO this has to be done here as it seems to be lost with reinitialize
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        # Calculate wake
        self.floris.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary? It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = self.get_plane_of_points(
            normal_vector="y",
            planar_coordinate=crossstream_dist,
        )

        # Compute the cutplane
        y_plane = CutPlane(df, x_resolution, z_resolution)

        # Reset the fi object back to the turbine grid configuration
        self.floris = Floris.from_dict(floris_dict)
        self.floris.flow_field.het_map = self.het_map

        # Run the simulation again for futher postprocessing (i.e. now we can get farm power)
        self.calculate_wake()

        return y_plane

    def check_wind_condition_for_viz(self, wd=None, ws=None):
        if len(wd) > 1 or len(wd) < 1:
            raise ValueError("Wind direction input must be of length 1 for visualization. Current length is {}.".format(len(wd)))

        if len(ws) > 1 or len(ws) < 1:
            raise ValueError("Wind speed input must be of length 1 for visualization. Current length is {}.".format(len(ws)))

    def get_turbine_powers(self) -> NDArrayFloat:
        """Calculates the power at each turbine in the windfarm.

        Returns:
            NDArrayFloat: [description]
        """
        turbine_powers = power(
            air_density=self.floris.flow_field.air_density,
            velocities=self.floris.flow_field.u,
            yaw_angle=self.floris.farm.yaw_angles,
            pP=self.floris.farm.pPs,
            power_interp=self.floris.farm.turbine_power_interps,
            turbine_type_map=self.floris.farm.turbine_type_map,
        )
        return turbine_powers

    def get_turbine_Cts(self) -> NDArrayFloat:
        turbine_Cts = Ct(
            velocities=self.floris.flow_field.u,
            yaw_angle=self.floris.farm.yaw_angles,
            fCt=self.floris.farm.turbine_fCts,
            turbine_type_map=self.floris.farm.turbine_type_map,
        )
        return turbine_Cts

    def get_turbine_ais(self) -> NDArrayFloat:
        turbine_ais = axial_induction(
            velocities=self.floris.flow_field.u,
            yaw_angle=self.floris.farm.yaw_angles,
            fCt=self.floris.farm.turbine_fCts,
            turbine_type_map=self.floris.farm.turbine_type_map,
        )
        return turbine_ais

    def get_turbine_average_velocities(self) -> NDArrayFloat:
        turbine_avg_vels = average_velocity(
            velocities=self.floris.flow_field.u,
        )
        return turbine_avg_vels

    def get_farm_power(
        self,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
        use_turbulence_correction=False,
    ):
        """
        Report wind plant power from instance of floris. Optionally includes
        uncertainty in wind direction and yaw position when determining power.
        Uncertainty is included by computing the mean wind farm power for a
        distribution of wind direction and yaw position deviations from the
        original wind direction and yaw angles.

        Args:
            include_unc (bool): When *True*, uncertainty in wind direction
                and/or yaw position is included when determining wind farm
                power. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw':
                1.75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations.
                Defaults to *False*.

        Returns:
            float: Sum of wind turbine powers.
        """
        # TODO: Turbulence correction used in the power calculation, but may not be in
        # the model yet
        # TODO: Turbines need a switch for using turbulence correction
        # TODO: Uncomment out the following two lines once the above are resolved
        # for turbine in self.floris.farm.turbines:
        #     turbine.use_turbulence_correction = use_turbulence_correction

        if include_unc:
            unc_pmfs = _generate_uncertainty_parameters(unc_options, unc_pmfs)

            # TODO: The original form of this is:
            # self.floris.farm.wind_map.input_direction[0], but it's unclear why we're
            # capping at just the first wind direction. Should this behavior be kept?
            # I'm unsure as to how the first wind direction is the original, so it could
            # just be a naming thing that's throwing me off....
            wd_orig = self.floris.flow_field.wind_directions

            yaw_angles = self.floris.farm.yaw_angles()
            self.reinitialize(wind_direction=wd_orig + unc_pmfs["wd_unc"])
            power_at_yaw = [
                self.get_farm_power_for_yaw_angle(yaw_angles + delta_yaw, no_wake=no_wake)
                for delta_yaw in unc_pmfs["yaw_unc"]
            ]
            mean_farm_power = unc_pmfs["wd_unc_pmf"] * unc_pmfs["yaw_unc_pmf"] * np.array(power_at_yaw)

            # reinitialize with original values
            self.reinitialize(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)
            return mean_farm_power

        turbine_powers = self.get_turbine_powers()
        # for i in range(self.floris.farm.n_turbines):
        #     avg_v_i = average_velocity(
        #         self.floris.flow_field.u,
        #         ix_filter=[i]
        #     )
        #     power_i = power(
        #         air_density=self.floris.flow_field.air_density,
        #         velocities=self.floris.flow_field.u,
        #         yaw_angle=self.floris.farm.yaw_angles,
        #         pP=self.floris.turbine.pP,
        #         power_interp=self.floris.turbine.power_interp,
        #         ix_filter=[i]
        #     )
        #     print(i, avg_v_i[0,0,0], power_i[0,0,0])
        # print( "total, MW", np.sum(turbine_powers[0,0]) / 1000000 )
        return np.sum(turbine_powers, axis=2)

    def get_farm_AEP(
        self,
        wd: NDArrayFloat | list[float],
        ws: NDArrayFloat | list[float],
        freq: NDArrayFloat | list[list[float]],
        yaw: NDArrayFloat | list[float] | None = None,
        limit_ws: bool = False,
        ws_limit_tol: float = 0.001,
        ws_cutout: float = 30.0,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, wind rose probability, and yaw offset. This can be computed for
        pre-determined wind direction and wind speed combinations, as was the case in
        FLORIS v2, or additionally, the unique wind directions, wind speeds, and their
        probabilities can be input.

        Args:
            wd (NDArrayFloat | list[float]): List or array of wind direction values.
                Either a unique list of wind directions can be used or the wind
                directions corresponding to a pre-computed set of combinations
                should be used.
            ws (NDArrayFloat | list[float]): List or array of wind speed values.
                Either a unique list of wind speeds can be used or the wind speeds
                corresponding to a pre-computed set of combinations should be used.
            freq (NDArrayFloat | list[list[float]]): Frequencies corresponding to either
                the pre-computed combinations of wind directions and wind speeds or the
                full wind rose with dimensions (N wind directions x N wind speeds).
            yaw (NDArrayFloat | list[float] | None, optional): List or array of yaw
                values if wake is steering implemented that correspond with the number
                of wind directions. Defaults to None.
            limit_ws (bool, optional): When *True*, detect wind speed when power
                reaches it's maximum value for a given wind direction. For all
                higher wind speeds, use last calculated value when below cut
                out. Defaults to False.
            ws_limit_tol (float, optional): Tolerance fraction for determining
                wind speed where power stops changing. If limit_ws is *True*,
                assume power remains constant up to cut out for wind speeds
                above the point where power changes less than ws_limit_tol of
                the previous power. Defaults to 0.001.
            ws_cutout (float, optional): Cut out wind speed (m/s). If limit_ws
                is *True*, assume power is zero for wind speeds greater than or
                equal to ws_cutout. Defaults to 30.0

        Returns:
            float: AEP for wind farm.
        """

        # # Convert the required inputs to arrays
        # wd = np.array(wd)
        # ws = np.array(ws)
        # freq = np.array(freq)

        # # Determine if the direction and speed inputs provided are a set of pre-determined
        # # combinations, and compute the full combination set if so, where the value
        # # in freq will be set to 0 if a combination was not in the original set to ensure
        # # it's not counted.
        # wd_unique = wd.unique()
        # ws_unique = ws.unique()
        # if np.array_equal(wd_unique, sorted(wd)) and np.array_equal(ws_unique, sorted(ws)):
        #     # Reshape the frequency input if required, and leave the unique inputs as-is
        #     if freq.shape != (wd_unique.size, ws_unique.size):
        #         freq = freq.reshape((wd_unique.size, ws_unique.size))
        # else:
        #     # Compute all the combinations
        #     wd_unique, ws_unique, freq, yaw = correct_for_all_combinations(wd, ws, freq, yaw)

        # # If the yaw input is still None, then create a None array as inputs
        # if yaw is None:
        #     N = wd_unique.size * ws_unique.size
        #     yaw = np.array([None] * N).reshape(wd_unique.size, ws_unique.size)
        # else:
        #     yaw = np.array(yaw)

        # filter out wind speeds beyond the cutoff, if necessary
        # if limit_ws:
        #     ix_ws_filter = ws_unique >= ws_cutout
        #     ws_unique = ws_unique[ix_ws_filter]
        #     freq = freq[:, ix_ws_filter]
        #     yaw = yaw[:, ix_ws_filter]

        # self.reinitialize(wind_direction=wd_unique, wind_speed=ws_unique, wind_rose_probability=freq)
        self.calculate_wake()
        farm_power = self.get_farm_power()  # TODO: Do we need to specify an axis since this is a sum?
        AEP = farm_power * freq * 8760
        return np.sum(AEP)

    def _calc_one_AEP_case(self, wd, ws, freq, yaw=None):
        self.reinitialize(wind_direction=[wd], wind_speed=[ws])
        self.calculate_wake(yaw_angles=yaw)
        return self.get_farm_power() * freq * 8760

    # def get_farm_AEP_parallel(
    #     self,
    #     wd: NDArrayFloat | list[float],
    #     ws: NDArrayFloat | list[float],
    #     freq: NDArrayFloat | list[list[float]],
    #     yaw: NDArrayFloat | list[float] | None = None,
    #     jobs=-1,
    # ):
    #     """
    #     Estimate annual energy production (AEP) for distributions of wind
    #     speed, wind direction and yaw offset with parallel computations on
    #     a single comptuer.

    #     # TODO: Update the docstrings and allow for the use of precomputed combinations
    #     as well as unique inputs that need to be computed. Same for the other AEPs

    #     Args:
    #         wd (iterable): List or array of wind direction values.
    #         ws (iterable): List or array of wind speed values.
    #         freq (iterable): Frequencies corresponding to wind direction and wind speed
    #             combinations in the wind rose with, shape (N wind directions x N wind speeds).
    #         yaw (iterable, optional): List or array of yaw values if wake is steering
    #             implemented, with shape (N wind directions). Defaults to None.
    #         jobs (int, optional): The number of jobs (cores) to use in the parallel
    #             computations.

    #     Returns:
    #         float: AEP for wind farm.
    #     """
    #     if jobs < -1:
    #         raise ValueError("Input 'jobs' cannot be negative.")
    #     if jobs == -1:
    #         jobs = int(np.ceil(cpu_count() * 0.8))
    #     if jobs > 0:
    #         jobs = min(jobs, cpu_count())
    #     if jobs > len(wd):
    #         jobs = len(wd)

    #     if yaw is None:
    #         yaw = [None] * len(wd)

    #     wd = np.array(wd)
    #     ws = np.array(ws)
    #     freq = np.array(freq)

    #     # Make one large list of arguments, then flatten and resort the nested tuples
    #     # to the correct ordering of self, wd, ws, freq, yaw
    #     global_arguments = list(zip(repeat(self), zip(wd, yaw), ws, freq.flatten()))
    #     # OR is this supposed to be all wind speeds for each wind direction?:
    #     # global_arguments = list(zip(repeat(self), zip(wd, yaw), repeat(ws), freq))
    #     # global_arguments = [(s, n[0], wspd, f, n[1]) for s, n, wspd, f in global_arguments]
    #     global_arguments = [(s, n[0][0], n[1], f, n[0][1]) for s, n, f in global_arguments]

    #     num_cases = wd.size * ws.size
    #     chunksize = int(np.ceil(num_cases / jobs))

    #     with Pool(jobs) as pool:
    #         opt = pool.starmap(global_calc_one_AEP_case, global_arguments, chunksize=chunksize)
    #         # add AEP to overall AEP

    #     return 0.0 + np.sum(opt)

    def calculate_AEP_wind_limit(self, num_turbines, x_spacing, start_ws, threshold):
        orig_layout_x = self.layout_x
        orig_layout_y = self.layout_y
        D = self.floris.farm.turbines[0].rotor_diameter

        self.reinitialize(
            layout_array=(
                [i * x_spacing * D for i in range(num_turbines)],
                [0.0] * num_turbines,
            ),
            wind_speed=start_ws,
        )
        self.calculate_wake()

        prev_power = 1.0
        cur_power = self.get_farm_power()
        ws = start_ws

        while np.abs(prev_power - cur_power) / prev_power > threshold:
            prev_power = cur_power
            ws += 0.2
            self.reinitialize(wind_speed=ws)
            self.calculate_wake()
            cur_power = self.get_farm_power()
        ws += 1.0

        self.reinitialize(layout_array=(orig_layout_x, orig_layout_y), wind_speed=ws)
        self.calculate_wake()
        self.max_power = self.get_farm_power()
        self.ws_limit = ws

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine x-coordinate.
        """
        return self.floris.farm.layout_x

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine y-coordinate.
        """
        return self.floris.farm.layout_y


    def get_turbine_layout(self, z=False):
        """
        Get turbine layout

        Args:
            z (bool): When *True*, return lists of x, y, and z coords,
            otherwise, return x and y only. Defaults to *False*.

        Returns:
            np.array: lists of x, y, and (optionally) z coordinates of
                      each turbine
        """
        xcoords, ycoords, zcoords = np.array([c.elements for c in self.floris.farm.coordinates]).T
        if z:
            return xcoords, ycoords, zcoords
        else:
            return xcoords, ycoords


def _convert_v24_dictionary_to_v3(dict_legacy):
    """
    Converts a v2.4 floris input dictionary file to a v3.0-compatible
    dictionary. See detailed instructions in the class 
    FlorisInterface_legacy_v24.

    Args:
        dict_legacy (dict): Input dictionary in legacy floris v2.4 format.

    Returns:
        dict_out (dict): Converted dictionary containing the floris input
        settings in v3.0-compatible format.
    """
    # Simple entries that can just be copied over
    dict_out = dict()  # Output dictionary
    dict_out["name"] = dict_legacy["name"] + " (auto-converted to v3)"
    dict_out["description"] = dict_legacy["description"]
    dict_out["floris_version"] = "v3.0 (converted from legacy format v2)"
    dict_out["logging"] = dict_legacy["logging"]

    dict_out["solver"] = {
        "type": "turbine_grid",
        "turbine_grid_points": dict_legacy["turbine"]["properties"]["ngrid"],
    }

    fp = dict_legacy["farm"]["properties"]
    tp = dict_legacy["turbine"]["properties"]
    dict_out["farm"] = {
        "layout_x": fp["layout_x"],
        "layout_y": fp["layout_y"],
    }

    ref_height = fp["specified_wind_height"]
    if ref_height < 0:
        ref_height = tp["hub_height"]

    dict_out["flow_field"] = {
        "air_density": fp["air_density"],
        "reference_wind_height": ref_height,
        "turbulence_intensity": fp["turbulence_intensity"][0],
        "wind_directions": [fp["wind_direction"]],
        "wind_shear": fp["wind_shear"],
        "wind_speeds": [fp["wind_speed"]],
        "wind_veer": fp["wind_veer"],
    }

    wp = dict_legacy["wake"]["properties"]
    velocity_model = wp["velocity_model"]
    velocity_model_str = velocity_model
    if velocity_model == "gauss_legacy":
        velocity_model_str = "gauss"
    deflection_model = wp["deflection_model"]
    turbulence_model = wp["turbulence_model"]
    wdp = wp["parameters"]["wake_deflection_parameters"][deflection_model]
    wvp = wp["parameters"]["wake_velocity_parameters"][velocity_model]
    wtp = wp["parameters"]["wake_turbulence_parameters"][turbulence_model]
    dict_out["wake"] = {
        "model_strings": {
            "combination_model": wp["combination_model"],
            "deflection_model": deflection_model,
            "turbulence_model": turbulence_model,
            "velocity_model": velocity_model_str,
        },
        "enable_secondary_steering": wdp["use_secondary_steering"],
        "enable_yaw_added_recovery": wvp["use_yaw_added_recovery"],
        "enable_transverse_velocities": wvp["calculate_VW_velocities"],
    }

    # Copy over wake velocity parameters and remove unnecessary parameters
    velocity_subdict = copy.deepcopy(wvp)
    c = ["calculate_VW_velocities", "use_yaw_added_recovery", "eps_gain"]
    for ci in [ci for ci in c if ci in velocity_subdict.keys()]:
        velocity_subdict.pop(ci)

    # Copy over wake deflection parameters and remove unnecessary parameters
    deflection_subdict = copy.deepcopy(wdp)
    c = ["use_secondary_steering"]
    for ci in [ci for ci in c if ci in deflection_subdict.keys()]:
        deflection_subdict.pop(ci)

    # Copy over wake turbulence parameters and remove unnecessary parameters
    turbulence_subdict = copy.deepcopy(wtp)

    # Save parameter settings to wake dictionary
    dict_out["wake"]["wake_velocity_parameters"] = {
        velocity_model_str: velocity_subdict
    }
    dict_out["wake"]["wake_deflection_parameters"] = {
        deflection_model: deflection_subdict
    }
    dict_out["wake"]["wake_turbulence_parameters"] = {
        turbulence_model: turbulence_subdict
    }

    # Finally add turbine information
    dict_out["turbine"] = {
        "generator_efficiency": tp["generator_efficiency"],
        "hub_height": tp["hub_height"],
        "pP": tp["pP"],
        "pT": tp["pT"],
        "rotor_diameter": tp["rotor_diameter"],
        "TSR": tp["TSR"],
        "power_thrust_table": tp["power_thrust_table"],
    }

    return dict_out


class FlorisInterface_legacy_v24(FlorisInterface):
    """
    FlorisInterface_legacy_v24 provides a wrapper around FlorisInterface
    which enables compatibility of the class with legacy floris v2.4 input
    files. The user can simply pass this class the path to a legacy v2.4
    floris input file to this class and it'll convert it to a v3.0-compatible
    input dictionary and load the floris v3.0 object.

    After successfully loading the v3.0 Floris object, you can export the
    input file using: fi.floris.to_file("converted_input_file_v3.yaml").

    If you would like to manually convert the input dictionary without first
    loading it in FLORIS, or if somehow the code fails to automatically
    convert the input file to v3, you should follow the following steps:
      1. Load the legacy v2.4 input floris JSON file as a dictionary
      2. Pass the v2.4 dictionary to `_convert_v24_dictionary_to_v3(...)`.
         That will return a v3.0-compatible input dictionary.
      3. Save the converted configuration file to a YAML or JSON file.

      For example:

        import json, yaml
        from floris.tools.floris_interface import _convert_v24_dictionary_to_v3

        with open(<path_to_legacy_v24_input_file.json>) as legacy_dict_file:
            configuration_v2 = json.load(legacy_dict_file)
        configuration_v3 = _convert_v24_dictionary_to_v3(configuration_v2)
        with open(r'converted_configuration_file_v3.yaml', 'w') as file:
            yaml.dump(configuration_v3, file)

    Args:
        configuration (:py:obj:`dict`): The legacy v2.4 Floris configuration
            dictionary or the file path to the JSON file.
    """

    def __init__(self, configuration: dict | str | Path, het_map=None):

        if not isinstance(configuration, (str, Path, dict)):
            raise TypeError("The Floris `configuration` must of type 'dict', 'str', or 'Path'.")

        print("Importing and converting legacy floris v2.4 input file...")
        if isinstance(configuration, (str, Path)):
            import json
            with open(configuration) as legacy_dict_file:
                configuration = json.load(legacy_dict_file)

        configuration = _convert_v24_dictionary_to_v3(configuration)
        super().__init__(configuration, het_map=het_map)  # Initialize full class


def generate_heterogeneous_wind_map(speed_ups, x, y, z=None):
    if z is not None:
        # Compute the 3-dimensional interpolants for each wind diretion
        # Linear interpolation is used for points within the user-defined area of values,
        # while a nearest-neighbor interpolant is used for points outside that region
        in_region = [LinearNDInterpolator(list(zip(x, y, z)), speed_up, fill_value=np.nan) for speed_up in speed_ups]
        out_region = [NearestNDInterpolator(list(zip(x, y, z)), speed_up) for speed_up in speed_ups]
    else:
        # Compute the 2-dimensional interpolants for each wind diretion
        # Linear interpolation is used for points within the user-defined area of values,
        # while a nearest-neighbor interpolant is used for points outside that region
        in_region = [LinearNDInterpolator(list(zip(x, y)), speed_up, fill_value=np.nan) for speed_up in speed_ups]
        out_region = [NearestNDInterpolator(list(zip(x, y)), speed_up) for speed_up in speed_ups]

    return [in_region, out_region]

# def global_calc_one_AEP_case(FlorisInterface, wd, ws, freq, yaw=None):
#     return FlorisInterface._calc_one_AEP_case(wd, ws, freq, yaw)

DEFAULT_UNCERTAINTY = {"std_wd": 4.95, "std_yaw": 1.75, "pmf_res": 1.0, "pdf_cutoff": 0.995}


def _generate_uncertainty_parameters(unc_options: dict, unc_pmfs: dict) -> dict:
    """Generates the uncertainty parameters for `FlorisInterface.get_farm_power` and
    `FlorisInterface.get_turbine_power` for more details.

    Args:
        unc_options (dict): See `FlorisInterface.get_farm_power` or `FlorisInterface.get_turbine_power`.
        unc_pmfs (dict): See `FlorisInterface.get_farm_power` or `FlorisInterface.get_turbine_power`.

    Returns:
        dict: [description]
    """
    if (unc_options is None) & (unc_pmfs is None):
        unc_options = DEFAULT_UNCERTAINTY

    if unc_pmfs is not None:
        return unc_pmfs

    wd_unc = np.zeros(1)
    wd_unc_pmf = np.ones(1)
    yaw_unc = np.zeros(1)
    yaw_unc_pmf = np.ones(1)

    # create normally distributed wd and yaw uncertaitny pmfs if appropriate
    if unc_options["std_wd"] > 0:
        wd_bnd = int(np.ceil(norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_wd"]) / unc_options["pmf_res"]))
        bound = wd_bnd * unc_options["pmf_res"]
        wd_unc = np.linspace(-1 * bound, bound, 2 * wd_bnd + 1)
        wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options["std_wd"])
        wd_unc_pmf /= np.sum(wd_unc_pmf)  # normalize so sum = 1.0

    if unc_options["std_yaw"] > 0:
        yaw_bnd = int(
            np.ceil(norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_yaw"]) / unc_options["pmf_res"])
        )
        bound = yaw_bnd * unc_options["pmf_res"]
        yaw_unc = np.linspace(-1 * bound, bound, 2 * yaw_bnd + 1)
        yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options["std_yaw"])
        yaw_unc_pmf /= np.sum(yaw_unc_pmf)  # normalize so sum = 1.0

    unc_pmfs = {
        "wd_unc": wd_unc,
        "wd_unc_pmf": wd_unc_pmf,
        "yaw_unc": yaw_unc,
        "yaw_unc_pmf": yaw_unc_pmf,
    }
    return unc_pmfs


# def correct_for_all_combinations(
#     wd: NDArrayFloat,
#     ws: NDArrayFloat,
#     freq: NDArrayFloat,
#     yaw: NDArrayFloat | None = None,
# ) -> tuple[NDArrayFloat]:
#     """Computes the probabilities for the complete windrose from the desired wind
#     direction and wind speed combinations and their associated probabilities so that
#     any undesired combinations are filled with a 0.0 probability.

#     Args:
#         wd (NDArrayFloat): List or array of wind direction values.
#         ws (NDArrayFloat): List or array of wind speed values.
#         freq (NDArrayFloat): Frequencies corresponding to wind
#             speeds and directions in wind rose with dimensions
#             (N wind directions x N wind speeds).
#         yaw (NDArrayFloat | None): The corresponding yaw angles for each of the wind
#             direction and wind speed combinations, or None. Defaults to None.

#     Returns:
#         NDArrayFloat, NDArrayFloat, NDArrayFloat: The unique wind directions, wind
#             speeds, and the associated probability of their combination combinations in
#             an array of shape (N wind directions x N wind speeds).
#     """

#     combos_to_compute = np.array(list(zip(wd, ws, freq)))

#     unique_wd = wd.unique()
#     unique_ws = ws.unique()
#     all_combos = np.array(list(product(unique_wd, unique_ws)), dtype=float)
#     all_combos = np.hstack((all_combos, np.zeros((all_combos.shape[0], 1), dtype=float)))
#     expanded_yaw = np.array([None] * all_combos.shape[0]).reshape(unique_wd.size, unique_ws.size)

#     ix_match = [np.where((all_combos[:, :2] == combo[:2]).all(1))[0][0] for combo in combos_to_compute]
#     all_combos[ix_match, 2] = combos_to_compute[:, 2]
#     if yaw is not None:
#         expanded_yaw[ix_match] = yaw
#     freq = all_combos.T[2].reshape((unique_wd.size, unique_ws.size))
#     return unique_wd, unique_ws, freq


    # def get_set_of_points(self, x_points, y_points, z_points):
    #     """
    #     Calculates velocity values through the
    #     :py:meth:`~.FlowField.calculate_wake` method at points specified by
    #     inputs.

    #     Args:
    #         x_points (float): X-locations to get velocity values at.
    #         y_points (float): Y-locations to get velocity values at.
    #         z_points (float): Z-locations to get velocity values at.

    #     Returns:
    #         :py:class:`pandas.DataFrame`: containing values of x, y, z, u, v, w
    #     """
    #     # Get a copy for the flow field so don't change underlying grid points
    #     flow_field = copy.deepcopy(self.floris.flow_field)

    #     if hasattr(self.floris.wake.velocity_model, "requires_resolution"):
    #         if self.floris.velocity_model.requires_resolution:

    #             # If this is a gridded model, must extract from full flow field
    #             self.logger.info(
    #                 "Model identified as %s requires use of underlying grid print"
    #                 % self.floris.wake.velocity_model.model_string
    #             )
    #             self.logger.warning("FUNCTION NOT AVAILABLE CURRENTLY")

    #     # Set up points matrix
    #     points = np.row_stack((x_points, y_points, z_points))

    #     # TODO: Calculate wake inputs need to be mapped
    #     raise_error = True
    #     if raise_error:
    #         raise NotImplementedError("Additional point calculation is not yet supported!")
    #     # Recalculate wake with these points
    #     flow_field.calculate_wake(points=points)

    #     # Get results vectors
    #     x_flat = flow_field.x.flatten()
    #     y_flat = flow_field.y.flatten()
    #     z_flat = flow_field.z.flatten()
    #     u_flat = flow_field.u.flatten()
    #     v_flat = flow_field.v.flatten()
    #     w_flat = flow_field.w.flatten()

    #     df = pd.DataFrame(
    #         {
    #             "x": x_flat,
    #             "y": y_flat,
    #             "z": z_flat,
    #             "u": u_flat,
    #             "v": v_flat,
    #             "w": w_flat,
    #         }
    #     )

    #     # Subset to points requests
    #     df = df[df.x.isin(x_points)]
    #     df = df[df.y.isin(y_points)]
    #     df = df[df.z.isin(z_points)]

    #     # Drop duplicates
    #     df = df.drop_duplicates()

    #     # Return the dataframe
    #     return df
    
    # def get_flow_data(self, resolution=None, grid_spacing=10, velocity_deficit=False):
    #     """
    #     Generate :py:class:`~.tools.flow_data.FlowData` object corresponding to
    #     active FLORIS instance.

    #     Velocity and wake models requiring calculation on a grid implement a
    #     discretized domain at resolution **grid_spacing**. This is distinct
    #     from the resolution of the returned flow field domain.

    #     Args:
    #         resolution (float, optional): Resolution of output data.
    #             Only used for wake models that require spatial
    #             resolution (e.g. curl). Defaults to None.
    #         grid_spacing (int, optional): Resolution of grid used for
    #             simulation. Model results may be sensitive to resolution.
    #             Defaults to 10.
    #         velocity_deficit (bool, optional): When *True*, normalizes velocity
    #             with respect to initial flow field velocity to show relative
    #             velocity deficit (%). Defaults to *False*.

    #     Returns:
    #         :py:class:`~.tools.flow_data.FlowData`: FlowData object
    #     """

    #     if resolution is None:
    #         if not self.floris.wake.velocity_model.requires_resolution:
    #             self.logger.info("Assuming grid with spacing %d" % grid_spacing)
    #             (
    #                 xmin,
    #                 xmax,
    #                 ymin,
    #                 ymax,
    #                 zmin,
    #                 zmax,
    #             ) = self.floris.flow_field.domain_bounds  # TODO: No grid attribute within FlowField
    #             resolution = Vec3(
    #                 1 + (xmax - xmin) / grid_spacing,
    #                 1 + (ymax - ymin) / grid_spacing,
    #                 1 + (zmax - zmin) / grid_spacing,
    #             )
    #         else:
    #             self.logger.info("Assuming model resolution")
    #             resolution = self.floris.wake.velocity_model.model_grid_resolution

    #     # Get a copy for the flow field so don't change underlying grid points
    #     flow_field = copy.deepcopy(self.floris.flow_field)

    #     if (
    #         flow_field.wake.velocity_model.requires_resolution
    #         and flow_field.wake.velocity_model.model_grid_resolution != resolution
    #     ):
    #         self.logger.warning(
    #             "WARNING: The current wake velocity model contains a "
    #             + "required grid resolution; the Resolution given to "
    #             + "FlorisInterface.get_flow_field is ignored."
    #         )
    #         resolution = flow_field.wake.velocity_model.model_grid_resolution
    #     flow_field.reinitialize(with_resolution=resolution)  # TODO: Not implemented
    #     self.logger.info(resolution)
    #     # print(resolution)
    #     flow_field.steady_state_atmospheric_condition()

    #     order = "f"
    #     x = flow_field.x.flatten(order=order)
    #     y = flow_field.y.flatten(order=order)
    #     z = flow_field.z.flatten(order=order)

    #     u = flow_field.u.flatten(order=order)
    #     v = flow_field.v.flatten(order=order)
    #     w = flow_field.w.flatten(order=order)

    #     # find percent velocity deficit
    #     if velocity_deficit:
    #         u = abs(u - flow_field.u_initial.flatten(order=order)) / flow_field.u_initial.flatten(order=order) * 100
    #         v = abs(v - flow_field.v_initial.flatten(order=order)) / flow_field.v_initial.flatten(order=order) * 100
    #         w = abs(w - flow_field.w_initial.flatten(order=order)) / flow_field.w_initial.flatten(order=order) * 100

    #     # Determine spacing, dimensions and origin
    #     unique_x = np.sort(np.unique(x))
    #     unique_y = np.sort(np.unique(y))
    #     unique_z = np.sort(np.unique(z))
    #     spacing = Vec3(
    #         unique_x[1] - unique_x[0],
    #         unique_y[1] - unique_y[0],
    #         unique_z[1] - unique_z[0],
    #     )
    #     dimensions = Vec3(len(unique_x), len(unique_y), len(unique_z))
    #     origin = Vec3(0.0, 0.0, 0.0)
    #     return FlowData(x, y, z, u, v, w, spacing=spacing, dimensions=dimensions, origin=origin)


    # def get_turbine_power(
    #     self,
    #     include_unc=False,
    #     unc_pmfs=None,
    #     unc_options=None,
    #     no_wake=False,
    #     use_turbulence_correction=False,
    # ):
    #     """
    #     Report power from each wind turbine.

    #     Args:
    #         include_unc (bool): If *True*, uncertainty in wind direction
    #             and/or yaw position is included when determining turbine
    #             powers. Defaults to *False*.
    #         unc_pmfs (dictionary, optional): A dictionary containing optional
    #             probability mass functions describing the distribution of wind
    #             direction and yaw position deviations when wind direction and/or
    #             yaw position uncertainty is included in the power calculations.
    #             Contains the following key-value pairs:

    #             -   **wd_unc** (*np.array*): Wind direction deviations from the
    #                 original wind direction.
    #             -   **wd_unc_pmf** (*np.array*): Probability of each wind
    #                 direction deviation in **wd_unc** occuring.
    #             -   **yaw_unc** (*np.array*): Yaw angle deviations from the
    #                 original yaw angles.
    #             -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
    #                 deviation in **yaw_unc** occuring.

    #             Defaults to None, in which case default PMFs are calculated
    #             using values provided in **unc_options**.
    #         unc_options (dictionary, optional): A dictionary containing values
    #             used to create normally-distributed, zero-mean probability mass
    #             functions describing the distribution of wind direction and yaw
    #             position deviations when wind direction and/or yaw position
    #             uncertainty is included. This argument is only used when
    #             **unc_pmfs** is None and contains the following key-value pairs:

    #             -   **std_wd** (*float*): A float containing the standard
    #                 deviation of the wind direction deviations from the
    #                 original wind direction.
    #             -   **std_yaw** (*float*): A float containing the standard
    #                 deviation of the yaw angle deviations from the original yaw
    #                 angles.
    #             -   **pmf_res** (*float*): A float containing the resolution in
    #                 degrees of the wind direction and yaw angle PMFs.
    #             -   **pdf_cutoff** (*float*): A float containing the cumulative
    #                 distribution function value at which the tails of the
    #                 PMFs are truncated.

    #             Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.
    #             75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
    #         no_wake: (bool, optional): When *True* updates the turbine
    #             quantities without calculating the wake or adding the
    #             wake to the flow field. Defaults to *False*.
    #         use_turbulence_correction: (bool, optional): When *True* uses a
    #             turbulence parameter to adjust power output calculations.
    #             Defaults to *False*.

    #     Returns:
    #         np.array: Power produced by each wind turbine.
    #     """
    #     # TODO: Turbulence correction used in the power calculation, but may not be in
    #     # the model yet
    #     # TODO: Turbines need a switch for using turbulence correction
    #     # TODO: Uncomment out the following two lines once the above are resolved
    #     # for turbine in self.floris.farm.turbines:
    #     #     turbine.use_turbulence_correction = use_turbulence_correction

    #     if include_unc:
    #         unc_pmfs = _generate_uncertainty_parameters(unc_options, unc_pmfs)

    #         mean_farm_power = np.zeros(self.floris.farm.n_turbines)
    #         wd_orig = self.floris.flow_field.wind_directions  # TODO: same comment as in get_farm_power

    #         yaw_angles = self.get_yaw_angles()
    #         self.reinitialize(wind_direction=wd_orig[0] + unc_pmfs["wd_unc"])
    #         for i, delta_yaw in enumerate(unc_pmfs["yaw_unc"]):
    #             self.calculate_wake(
    #                 yaw_angles=list(np.array(yaw_angles) + delta_yaw),
    #                 no_wake=no_wake,
    #             )
    #             mean_farm_power += unc_pmfs["wd_unc_pmf"] * unc_pmfs["yaw_unc_pmf"][i] * self._get_turbine_powers()

    #         # reinitialize with original values
    #         self.reinitialize(wind_direction=wd_orig)
    #         self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)
    #         return mean_farm_power

    #     return self._get_turbine_powers()

    # def get_power_curve(self, wind_speeds):
    #     """
    #     Return the power curve given a set of wind speeds

    #     Args:
    #         wind_speeds (np.array): array of wind speeds to get power curve
    #     """

    #     # TODO: Why is this done? Should we expand for evenutal multiple turbines types
    #     # or just allow a filter on the turbine index?
    #     # Temporarily set the farm to a single turbine
    #     saved_layout_x = self.layout_x
    #     saved_layout_y = self.layout_y

    #     self.reinitialize(wind_speed=wind_speeds, layout_array=([0], [0]))
    #     self.calculate_wake()
    #     turbine_power = self._get_turbine_powers()

    #     # Set it back
    #     self.reinitialize(layout_array=(saved_layout_x, saved_layout_y))

    #     return turbine_power

    # def get_farm_power_for_yaw_angle(
    #     self,
    #     yaw_angles,
    #     include_unc=False,
    #     unc_pmfs=None,
    #     unc_options=None,
    #     no_wake=False,
    # ):
    #     """
    #     Assign yaw angles to turbines, calculate wake, and report farm power.

    #     Args:
    #         yaw_angles (np.array): Yaw to apply to each turbine.
    #         include_unc (bool, optional): When *True*, includes wind direction
    #             uncertainty in estimate of wind farm power. Defaults to *False*.
    #         unc_pmfs (dictionary, optional): A dictionary containing optional
    #             probability mass functions describing the distribution of wind
    #             direction and yaw position deviations when wind direction and/or
    #             yaw position uncertainty is included in the power calculations.
    #             Contains the following key-value pairs:

    #             -   **wd_unc** (*np.array*): Wind direction deviations from the
    #                 original wind direction.
    #             -   **wd_unc_pmf** (*np.array*): Probability of each wind
    #                 direction deviation in **wd_unc** occuring.
    #             -   **yaw_unc** (*np.array*): Yaw angle deviations from the
    #                 original yaw angles.
    #             -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
    #                 deviation in **yaw_unc** occuring.

    #             Defaults to None, in which case default PMFs are calculated
    #             using values provided in **unc_options**.
    #         unc_options (dictionary, optional): A dictionary containing values
    #             used to create normally-distributed, zero-mean probability mass
    #             functions describing the distribution of wind direction and yaw
    #             position deviations when wind direction and/or yaw position
    #             uncertainty is included. This argument is only used when
    #             **unc_pmfs** is None and contains the following key-value pairs:

    #             -   **std_wd** (*float*): A float containing the standard
    #                 deviation of the wind direction deviations from the
    #                 original wind direction.
    #             -   **std_yaw** (*float*): A float containing the standard
    #                 deviation of the yaw angle deviations from the original yaw
    #                 angles.
    #             -   **pmf_res** (*float*): A float containing the resolution in
    #                 degrees of the wind direction and yaw angle PMFs.
    #             -   **pdf_cutoff** (*float*): A float containing the cumulative
    #                 distribution function value at which the tails of the
    #                 PMFs are truncated.

    #             Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.
    #             75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
    #         no_wake: (bool, optional): When *True* updates the turbine
    #             quantities without calculating the wake or adding the
    #             wake to the flow field. Defaults to *False*.

    #     Returns:
    #         float: Wind plant power. #TODO negative? in kW?
    #     """

    #     self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)

    #     return self.get_farm_power(include_unc=include_unc, unc_pmfs=unc_pmfs, unc_options=unc_options)

    # def copy_and_update_turbine_map(
    #     self, base_turbine_id: str, update_parameters: dict, new_id: str | None = None
    # ) -> dict:
    #     """Creates a new copy of an existing turbine and updates the parameters based on
    #     user input. This function is a helper to make the v2 -> v3 transition easier.

    #     Args:
    #         base_turbine_id (str): The base turbine's ID in `floris.farm.turbine_id`.
    #         update_parameters (dict): A dictionary of the turbine parameters to update
    #             and their new valies.
    #         new_id (str, optional): The new `turbine_id`, if `None` a unique
    #             identifier will be appended to the end. Defaults to None.

    #     Returns:
    #         dict: A turbine mapping that can be passed directly to `change_turbine`.
    #     """
    #     if new_id is None:
    #         new_id = f"{base_turbine_id}_copy{self.unique_copy_id}"
    #         self.unique_copy_id += 1

    #     turbine = {new_id: self.floris.turbine[base_turbine_id]._asdict()}
    #     turbine[new_id].update(update_parameters)
    #     return turbine

    # def change_turbine(
    #     self,
    #     turbine_indices: list[int],
    #     new_turbine_map: dict[str, dict[str, Any]],
    #     update_specified_wind_height: bool = False,
    # ):
    #     """
    #     Change turbine properties for specified turbines.

    #     Args:
    #         turbine_indices (list[int]): List of turbine indices to change.
    #         new_turbine_map (dict[str, dict[str, Any]]): New dictionary of turbine
    #             parameters to create the new turbines for each of `turbine_indices`.
    #         update_specified_wind_height (bool, optional): When *True*, update specified
    #             wind height to match new hub_height. Defaults to *False*.
    #     """
    #     new_turbine = True
    #     new_turbine_id = [*new_turbine_map][0]
    #     if new_turbine_id in self.floris.farm.turbine_map:
    #         new_turbine = False
    #         self.logger.info(f"Turbines {turbine_indices} will be re-mapped to the definition for: {new_turbine_id}")

    #     self.floris.farm.turbine_id = [
    #         new_turbine_id if i in turbine_indices else t_id for i, t_id in enumerate(self.floris.farm.turbine_id)
    #     ]
    #     if new_turbine:
    #         self.logger.info(f"Turbines {turbine_indices} have been mapped to the new definition for: {new_turbine_id}")

    #     # Update the turbine mapping if a new turbine was provided, then regenerate the
    #     # farm arrays for the turbine farm
    #     if new_turbine:
    #         turbine_map = self.floris.farm._asdict()["turbine_map"]
    #         turbine_map.update(new_turbine_map)
    #         self.floris.farm.turbine_map = turbine_map
    #     self.floris.farm.generate_farm_points()

    #     new_hub_height = new_turbine_map[new_turbine_id]["hub_height"]
    #     changed_hub_height = new_hub_height != self.floris.flow_field.reference_wind_height

    #     # Alert user if changing hub-height and not specified wind height
    #     if changed_hub_height and not update_specified_wind_height:
    #         self.logger.info("Note, updating hub height but not updating " + "the specfied_wind_height")

    #     if changed_hub_height and update_specified_wind_height:
    #         self.logger.info(f"Note, specfied_wind_height changed to hub-height: {new_hub_height}")
    #         self.reinitialize(specified_wind_height=new_hub_height)

    #     # Finish by re-initalizing the flow field
    #     self.reinitialize()

    # def set_use_points_on_perimeter(self, use_points_on_perimeter=False):
    #     """
    #     Set whether to use the points on the rotor diameter (perimeter) when
    #     calculating flow field and wake.

    #     Args:
    #         use_points_on_perimeter (bool): When *True*, use points at rotor
    #             perimeter in wake and flow calculations. Defaults to *False*.
    #     """
    #     for turbine in self.floris.farm.turbines:
    #         turbine.use_points_on_perimeter = use_points_on_perimeter
    #         turbine.initialize_turbine()

    # def set_gch(self, enable=True):
    #     """
    #     Enable or disable Gauss-Curl Hybrid (GCH) functions
    #     :py:meth:`~.GaussianModel.calculate_VW`,
    #     :py:meth:`~.GaussianModel.yaw_added_recovery_correction`, and
    #     :py:attr:`~.VelocityDeflection.use_secondary_steering`.

    #     Args:
    #         enable (bool, optional): Flag whether or not to implement flow
    #             corrections from GCH model. Defaults to *True*.
    #     """
    #     self.set_gch_yaw_added_recovery(enable)
    #     self.set_gch_secondary_steering(enable)

    # def set_gch_yaw_added_recovery(self, enable=True):
    #     """
    #     Enable or Disable yaw-added recovery (YAR) from the Gauss-Curl Hybrid
    #     (GCH) model and the control state of
    #     :py:meth:`~.GaussianModel.calculate_VW_velocities` and
    #     :py:meth:`~.GaussianModel.yaw_added_recovery_correction`.

    #     Args:
    #         enable (bool, optional): Flag whether or not to implement yaw-added
    #             recovery from GCH model. Defaults to *True*.
    #     """
    #     model_params = self.get_model_parameters()
    #     use_secondary_steering = model_params["Wake Deflection Parameters"]["use_secondary_steering"]

    #     if enable:
    #         model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"] = True

    #         # If enabling be sure calc vw is on
    #         model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = True

    #     if not enable:
    #         model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"] = False

    #         # If secondary steering is also off, disable calculate_VW_velocities
    #         if not use_secondary_steering:
    #             model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = False

    #     self.set_model_parameters(model_params)
    #     self.reinitialize()

    # def set_gch_secondary_steering(self, enable=True):
    #     """
    #     Enable or Disable secondary steering (SS) from the Gauss-Curl Hybrid
    #     (GCH) model and the control state of
    #     :py:meth:`~.GaussianModel.calculate_VW_velocities` and
    #     :py:attr:`~.VelocityDeflection.use_secondary_steering`.

    #     Args:
    #         enable (bool, optional): Flag whether or not to implement secondary
    #         steering from GCH model. Defaults to *True*.
    #     """
    #     model_params = self.get_model_parameters()
    #     use_yaw_added_recovery = model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"]

    #     if enable:
    #         model_params["Wake Deflection Parameters"]["use_secondary_steering"] = True

    #         # If enabling be sure calc vw is on
    #         model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = True

    #     if not enable:
    #         model_params["Wake Deflection Parameters"]["use_secondary_steering"] = False

    #         # If yar is also off, disable calculate_VW_velocities
    #         if not use_yaw_added_recovery:
    #             model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = False

    #     self.set_model_parameters(model_params)
    #     self.reinitialize()

    # def show_model_parameters(
    #     self,
    #     params=None,
    #     verbose=False,
    #     wake_velocity_model=True,
    #     wake_deflection_model=True,
    #     turbulence_model=False,
    # ):
    #     """
    #     Helper function to print the current wake model parameters and values.
    #     Shortcut to :py:meth:`~.tools.interface_utilities.show_params`.

    #     Args:
    #         params (list, optional): Specific model parameters to be returned,
    #             supplied as a list of strings. If None, then returns all
    #             parameters. Defaults to None.
    #         verbose (bool, optional): If set to *True*, will return the
    #             docstrings for each parameter. Defaults to *False*.
    #         wake_velocity_model (bool, optional): If set to *True*, will return
    #             parameters from the wake_velocity model. If set to *False*, will
    #             exclude parameters from the wake velocity model. Defaults to
    #             *True*.
    #         wake_deflection_model (bool, optional): If set to *True*, will
    #             return parameters from the wake deflection model. If set to
    #             *False*, will exclude parameters from the wake deflection
    #             model. Defaults to *True*.
    #         turbulence_model (bool, optional): If set to *True*, will return
    #             parameters from the wake turbulence model. If set to *False*,
    #             will exclude parameters from the wake turbulence model.
    #             Defaults to *True*.
    #     """
    #     show_params(
    #         self.floris.wake,
    #         params,
    #         verbose,
    #         wake_velocity_model,
    #         wake_deflection_model,
    #         turbulence_model,
    #     )

    # def get_model_parameters(
    #     self,
    #     params=None,
    #     wake_velocity_model=True,
    #     wake_deflection_model=True,
    #     turbulence_model=True,
    # ):
    #     """
    #     Helper function to return the current wake model parameters and values.
    #     Shortcut to :py:meth:`~.tools.interface_utilities.get_params`.

    #     Args:
    #         params (list, optional): Specific model parameters to be returned,
    #             supplied as a list of strings. If None, then returns all
    #             parameters. Defaults to None.
    #         wake_velocity_model (bool, optional): If set to *True*, will return
    #             parameters from the wake_velocity model. If set to *False*, will
    #             exclude parameters from the wake velocity model. Defaults to
    #             *True*.
    #         wake_deflection_model (bool, optional): If set to *True*, will
    #             return parameters from the wake deflection model. If set to
    #             *False*, will exclude parameters from the wake deflection
    #             model. Defaults to *True*.
    #         turbulence_model ([type], optional): If set to *True*, will return
    #             parameters from the wake turbulence model. If set to *False*,
    #             will exclude parameters from the wake turbulence model.
    #             Defaults to *True*.

    #     Returns:
    #         dict: Dictionary containing model parameters and their values.
    #     """
    #     model_params = get_params(
    #         self.floris.wake, params, wake_velocity_model, wake_deflection_model, turbulence_model
    #     )

    #     return model_params

    # def set_model_parameters(self, params, verbose=True):
    #     """
    #     Helper function to set current wake model parameters.
    #     Shortcut to :py:meth:`~.tools.interface_utilities.set_params`.

    #     Args:
    #         params (dict): Specific model parameters to be set, supplied as a
    #             dictionary of key:value pairs.
    #         verbose (bool, optional): If set to *True*, will print information
    #             about each model parameter that is changed. Defaults to *True*.
    #     """
    #     self.floris.wake = set_params(self.floris.wake, params, verbose)






    # def vis_layout(
    #     self,
    #     ax=None,
    #     show_wake_lines=False,
    #     limit_dist=None,
    #     turbine_face_north=False,
    #     one_index_turbine=False,
    #     black_and_white=False,
    # ):
    #     """
    #     Visualize the layout of the wind farm in the floris instance.
    #     Shortcut to :py:meth:`~.tools.layout_functions.visualize_layout`.

    #     Args:
    #         ax (:py:class:`matplotlib.pyplot.axes`, optional):
    #             Figure axes. Defaults to None.
    #         show_wake_lines (bool, optional): Flag to control plotting of
    #             wake boundaries. Defaults to False.
    #         limit_dist (float, optional): Downstream limit to plot wakes.
    #             Defaults to None.
    #         turbine_face_north (bool, optional): Force orientation of wind
    #             turbines. Defaults to False.
    #         one_index_turbine (bool, optional): If *True*, 1st turbine is
    #             turbine 1.
    #     """
    #     for i, turbine in enumerate(self.floris.farm.turbines):
    #         D = turbine.rotor_diameter
    #         break
    #     layout_x, layout_y = self.get_turbine_layout()

    #     turbineLoc = build_turbine_loc(layout_x, layout_y)

    #     # Show visualize the turbine layout
    #     visualize_layout(
    #         turbineLoc,
    #         D,
    #         ax=ax,
    #         show_wake_lines=show_wake_lines,
    #         limit_dist=limit_dist,
    #         turbine_face_north=turbine_face_north,
    #         one_index_turbine=one_index_turbine,
    #         black_and_white=black_and_white,
    #     )

    # def show_flow_field(self, ax=None):
    #     """
    #     Shortcut method to
    #     :py:meth:`~.tools.visualization.visualize_cut_plane`.

    #     Args:
    #         ax (:py:class:`matplotlib.pyplot.axes` optional):
    #             Figure axes. Defaults to None.
    #     """
    #     # Get horizontal plane at default height (hub-height)
    #     hor_plane = self.get_hor_plane()

    #     # Plot and show
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     visualize_cut_plane(hor_plane, ax=ax)
    #     plt.show()



    ## Functionality removed in v3

    def set_rotor_diameter(self, rotor_diameter):
        """
        This function has been replaced and no longer works correctly, assigning an error
        """
        raise Exception(
            "FlorinInterface.set_rotor_diameter has been removed in favor of FlorinInterface.change_turbine. See examples/change_turbine/."
        )
