from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from attrs import define, field

from floris import logging_manager
from floris.core import (
    BaseClass,
    BaseLibrary,
    cc_solver,
    Farm,
    FlowField,
    FlowFieldPlanarGrid,
    full_flow_cc_solver,
    full_flow_sequential_solver,
    full_flow_turbopark_solver,
    Grid,
    PointsGrid,
    sequential_solver,
    State,
    TurbineCubatureGrid,
    TurbineGrid,
    turbopark_solver,
    WakeModelManager,
)
from floris.core.wake_model import (
    EmpiricalGauss,
    Gauss,
    JensenJimenez,
    NoneWake,
)
from floris.type_dec import NDArrayFloat
from floris.utilities import (
    load_yaml,
    reverse_rotate_coordinates_rel_west,
)


@define
class Core(BaseClass):
    """
    Top-level class that describes a Floris model and initializes the
    simulation. Use the :py:class:`~.simulation.farm.Farm` attribute to
    access other objects within the model.
    """

    logging: dict = field(converter=dict)
    solver: dict = field(converter=dict)
    wake: WakeModelManager = field(converter=WakeModelManager.from_dict)
    farm: Farm = field(converter=Farm.from_dict)
    flow_field: FlowField = field(converter=FlowField.from_dict)

    # These fields are included to appease the requirement that all inputs must
    # be mapped to a field in the class. They are not used in FLORIS.
    name: str  = field(converter=str)
    description: str = field(converter=str)
    floris_version: str = field(converter=str)

    grid: Grid | TurbineGrid | TurbineCubatureGrid | FlowFieldPlanarGrid | PointsGrid = field(
        init=False
    )

    def __attrs_post_init__(self) -> None:

        # Configure logging
        logging_manager.configure_console_log(
            self.logging["console"]["enable"],
            self.logging["console"]["level"],
        )
        logging_manager.configure_file_log(
            self.logging["file"]["enable"],
            self.logging["file"]["level"],
        )

        # Initialize farm quantities that depend on other objects
        self.farm.set_control_setpoints_to_reference(self.flow_field.n_findex)

        if self.solver["type"] == "turbine_grid":
            self.grid = TurbineGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                grid_resolution=self.solver["turbine_grid_points"],
            )
        elif self.solver["type"] == "turbine_cubature_grid":
            self.grid = TurbineCubatureGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                grid_resolution=self.solver["turbine_grid_points"],
            )
        elif self.solver["type"] == "flow_field_planar_grid":
            self.grid = FlowFieldPlanarGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                normal_vector=self.solver["normal_vector"],
                planar_coordinate=self.solver["planar_coordinate"],
                grid_resolution=self.solver["flow_field_grid_points"],
                x1_bounds=self.solver["flow_field_bounds"][0],
                x2_bounds=self.solver["flow_field_bounds"][1],
            )
        else:
            raise ValueError(
                "Supported solver types are "
                "[turbine_grid, turbine_cubature_grid, flow_field_grid, flow_field_planar_grid], "
                f"but type given was {self.solver['type']}"
            )

        if isinstance(self.grid, (TurbineGrid, TurbineCubatureGrid)):
            self.farm.set_sorted_indices(self.grid.sorted_coord_indices)
            self.farm.construct_turbine_type_map()

        if isinstance(self.wake.user_defined_wake_model, dict):
            self.wake.user_defined_wake_model = BaseLibrary.from_dict(
                self.wake.user_defined_wake_model
            )

    def initialize_domain(self):
        """Initialize solution space prior to wake calculations"""

        # Initialize field quantities; doing this immediately prior to doing
        # the calculation step allows for manipulating inputs in a script
        # without changing the data structures
        self.flow_field.initialize_velocity_field(self.grid)

        # Initialize farm quantities
        self.farm.initialize()

        self.state.INITIALIZED

    def solve_for_turbines(self):
        """Perform the steady-state wind farm wake calculations. Note that
        initialize_domain() is required to be called before this function."""

        if self.wake.user_defined_wake_model is not None:
            vel_model = "user_defined"
        else:
            vel_model = self.wake.model_strings["velocity_model"]
        model_parameters = _temp_create_single_wake_model_dict(self.wake, vel_model)

        if vel_model not in ["empirical_gauss", "user_defined"] and \
            any(t.correct_cp_ct_for_tilt for t in self.farm.turbines):
            self.logger.warning(
                "The current model does not account for vertical wake deflection due to " +
                "tilt. Corrections to power and thrust coefficient can be included, but no " +
                "vertical wake deflection will occur."
            )

        operation_model_awc = False
        for t in self.farm.turbines:
            if t.operation_model == "awc":
                operation_model_awc = True
        if vel_model != "empirical_gauss" and operation_model_awc:
            self.logger.warning(
                f"The current model `{vel_model}` does not account for additional wake mixing " +
                "due to active wake control. Corrections to power and thrust coefficient can " +
                "be included, but no enhanced wake recovery will occur."
            )

        if self.wake.user_defined_wake_model is not None:
            self.wake.user_defined_wake_model.turbine_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="cc":
            cc_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        elif vel_model=="turbopark":
            self.logger.warning(
                "The turbopark model has been superseded by the turboparkgauss model. We " +
                "recommend using `velocity_model: turboparkgauss` instead."
            )
            turbopark_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        elif vel_model=="empirical_gauss":
            model = EmpiricalGauss(**model_parameters)
            model.turbine_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="jensen":
            model = JensenJimenez(**model_parameters)
            model.turbine_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="gauss":
            model = Gauss(**model_parameters)
            model.turbine_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="none":
            model = NoneWake(**model_parameters)
            model.turbine_solve(self.farm, self.flow_field, self.grid)
        else:
            sequential_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )

        self.finalize()

    def solve_for_viz(self):
        # Do the calculation with the TurbineGrid for a single wind speed
        # and wind direction and 1 point on the grid. Then, use the result
        # to construct the full flow field grid.
        # This function call should be for a single wind direction and wind speed
        # since the memory consumption is very large.

        self.flow_field.initialize_velocity_field(self.grid)

        vel_model = self.wake.model_strings["velocity_model"]
        model_parameters = _temp_create_single_wake_model_dict(self.wake, vel_model)

        if self.wake.user_defined_wake_model is not None:
            self.wake.user_defined_wake_model.point_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="cc":
            full_flow_cc_solver(self.farm, self.flow_field, self.grid, self.wake)
        elif vel_model=="turbopark":
            full_flow_turbopark_solver(self.farm, self.flow_field, self.grid, self.wake)
        elif vel_model=="empirical_gauss":
            model = EmpiricalGauss(**model_parameters)
            model.point_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="jensen":
            model = JensenJimenez(**model_parameters)
            model.point_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="gauss":
            model = Gauss(**model_parameters)
            model.point_solve(self.farm, self.flow_field, self.grid)
        elif vel_model=="none":
            model = NoneWake(**model_parameters)
            model.point_solve(self.farm, self.flow_field, self.grid)
        else:
            full_flow_sequential_solver(self.farm, self.flow_field, self.grid, self.wake)

    def solve_for_points(self, x, y, z):
        # Do the calculation with the TurbineGrid for a single wind speed
        # and wind direction and a 3x3 rotor grid. Then, use the result
        # to construct the full flow field grid.
        # This function call should be for a single wind direction and wind speed
        # since the memory consumption is very large.

        # Instantiate the flow_grid
        field_grid = PointsGrid(
            points_x=x,
            points_y=y,
            points_z=z,
            turbine_coordinates=self.farm.coordinates,
            turbine_diameters=self.farm.rotor_diameters,
            wind_directions=self.flow_field.wind_directions,
            grid_resolution=1,
            x_center_of_rotation=self.grid.x_center_of_rotation,
            y_center_of_rotation=self.grid.y_center_of_rotation
        )

        self.flow_field.initialize_velocity_field(field_grid)

        vel_model = self.wake.model_strings["velocity_model"]
        model_parameters = _temp_create_single_wake_model_dict(self.wake, vel_model)

        if self.wake.user_defined_wake_model is not None:
            self.wake.user_defined_wake_model.point_solve(self.farm, self.flow_field, field_grid)
        elif vel_model=="cc":
            full_flow_cc_solver(self.farm, self.flow_field, field_grid, self.wake)
        elif vel_model=="turbopark":
            full_flow_turbopark_solver(self.farm, self.flow_field, field_grid, self.wake)
        elif vel_model=="empirical_gauss":
            model = EmpiricalGauss(**model_parameters)
            model.point_solve(self.farm, self.flow_field, field_grid)
        elif vel_model=="jensen":
            model = JensenJimenez(**model_parameters)
            model.point_solve(self.farm, self.flow_field, field_grid)
        elif vel_model=="gauss":
            model = Gauss(**model_parameters)
            model.point_solve(self.farm, self.flow_field, field_grid)
        elif vel_model=="none":
            model = NoneWake(**model_parameters)
            model.point_solve(self.farm, self.flow_field, field_grid)
        else:
            full_flow_sequential_solver(self.farm, self.flow_field, field_grid, self.wake)

        return self.flow_field.u_sorted[:,:,0,0] # Remove turbine grid dimensions

    def solve_for_velocity_deficit_profiles(
        self,
        direction: str,
        downstream_dists: NDArrayFloat | list,
        profile_range: NDArrayFloat | list,
        resolution: int,
        homogeneous_wind_speed: float,
        ref_rotor_diameter: float,
        x_start: float,
        y_start: float,
        reference_height: float,
    ) -> list[pd.DataFrame]:
        """
        Extract velocity deficit profiles. See
        :py:meth:`~floris.floris_model.FlorisModel.sample_velocity_deficit_profiles`
        for more details.
        """

        self.logger.warning(
            "Velocity deficit profiles will move to a Numpy data structure in the next release. "
            "See https://github.com/NatLabRockies/floris/pull/1194."
        )

        # Create a grid that contains coordinates for all the sample points in all profiles.
        # Effectively, this is a grid of parallel lines.
        n_lines = len(downstream_dists)

        # Coordinate system (x1, x2, x3) is used to define the sample points. The origin is at
        # (x_start, y_start, reference_height) and x1 is in the streamwise direction.
        # The x1-coordinate is fixed for every line (every row in  `x1`).
        x1 = np.atleast_2d(downstream_dists).T * np.ones((n_lines, resolution))

        if resolution == 1:
            single_line = [0.0]
        else:
            single_line = np.linspace(profile_range[0], profile_range[1], resolution)

        if direction == 'cross-stream':
            x2 = single_line * np.ones((n_lines, resolution))
            x3 = np.zeros((n_lines, resolution))
        elif direction == 'vertical':
            x3 = single_line * np.ones((n_lines, resolution))
            x2 = np.zeros((n_lines, resolution))

        # Find the coordinates of the sample points in the inertial frame (x, y, z). This is done
        # through one rotation and one translation.
        x, y, z = reverse_rotate_coordinates_rel_west(
            self.flow_field.wind_directions,
            x1[None, :, :],
            x2[None, :, :],
            x3[None, :, :],
            x_center_of_rotation=0.0,
            y_center_of_rotation=0.0,
        )
        x = np.squeeze(x, axis=0) + x_start
        y = np.squeeze(y, axis=0) + y_start
        z = np.squeeze(z, axis=0) + reference_height

        u = self.solve_for_points(x.flatten(), y.flatten(), z.flatten())
        u = np.reshape(u[0, :], (n_lines, resolution))
        velocity_deficit = (homogeneous_wind_speed - u) / homogeneous_wind_speed

        velocity_deficit_profiles = []

        for i in range(n_lines):
            df = pd.DataFrame(
                {
                    'x': x[i],
                    'y': y[i],
                    'z': z[i],
                    'x1/D': x1[i]/ref_rotor_diameter,
                    'x2/D': x2[i]/ref_rotor_diameter,
                    'x3/D': x3[i]/ref_rotor_diameter,
                    'velocity_deficit': velocity_deficit[i],
                }
            )
            velocity_deficit_profiles.append(df)

        return velocity_deficit_profiles

    def finalize(self):
        # Once the wake calculation is finished, unsort the values to match
        # the user-supplied order of things.
        self.flow_field.finalize(self.grid.unsorted_indices)
        self.farm.finalize()
        self.state = State.USED

    ## I/O

    @classmethod
    def from_file(cls, input_file_path: str | Path) -> Core:
        """Creates a `Floris` instance from an input file. Must be filetype YAML.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                input file.

        Returns:
            Floris: The class object instance.
        """
        input_dict = load_yaml(Path(input_file_path).resolve())
        check_input_file_for_v3_keys(input_dict)
        return Core.from_dict(input_dict)

    def to_file(self, output_file_path: str) -> None:
        """Converts the `Floris` object to an input-ready YAML file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the file.
        """
        with open(output_file_path, "w+") as f:
            yaml.dump(
                self.as_dict(),
                f,
                sort_keys=False,
                default_flow_style=False
            )

def check_input_file_for_v3_keys(input_dict) -> None:
    """
    Checks if any FLORIS v3 keys are present in the input file and raises special errors if
    the extra keys belong to a v3 definition of the input_dct.
    and raises special errors if the extra arguments belong to a v3 definition of the class.

    Args:
        input_dict (dict): The input dictionary to be checked for v3 keys.
    """
    v3_deprecation_msg = (
        "Consider using the convert_floris_input_v3_to_v4.py utility in floris/tools "
        "to convert from a FLORIS v3 input file to FLORIS v4. "
        "See https://natlabrockies.github.io/floris/upgrade_guides/v3_to_v4.html "
        "for more information."
    )
    if "turbulence_intensity" in input_dict["flow_field"]:
        raise AttributeError(
            "turbulence_intensity has been updated to turbulence_intensities in FLORIS v4. "
            + v3_deprecation_msg
        )
    elif not hasattr(input_dict["flow_field"]["turbulence_intensities"], "__len__"):
        raise AttributeError(
            "turbulence_intensities must be a list of floats in FLORIS v4. "
            + v3_deprecation_msg
        )

    if input_dict["wake"]["model_strings"]["velocity_model"] == "multidim_cp_ct":
        raise AttributeError(
            "Dedicated 'multidim_cp_ct' velocity model has been removed in FLORIS v4 in favor of "
            + "supporting all available wake models. To recover previous operation, set "
            + "velocity_model to gauss. "
            + v3_deprecation_msg
        )

def _temp_create_single_wake_model_dict(wake, vel_model):
    """
    This is a temporary function until wake model parametrization is unified on the
    input dictionary. However, we may use it going forward for back compatibility with
    v4. In that case, checks should be made to ensure compatible deficit/deflection/turbulence
    models are being used together.
    """
    if vel_model == "gauss":
        model_parameters = wake.wake_velocity_parameters["gauss"] | \
            wake.wake_deflection_parameters["gauss"] | \
            wake.wake_turbulence_parameters["crespo_hernandez"]
        model_parameters["enable_transverse_velocities"] = wake.enable_transverse_velocities
        model_parameters["enable_yaw_added_recovery"] = wake.enable_yaw_added_recovery
        model_parameters["enable_secondary_steering"] = wake.enable_secondary_steering
    elif vel_model == "jensen":
        model_parameters = wake.wake_velocity_parameters["jensen"] | \
            wake.wake_deflection_parameters["jimenez"] | \
            wake.wake_turbulence_parameters["crespo_hernandez"]
    elif vel_model == "empirical_gauss":
        model_parameters = wake.wake_velocity_parameters["empirical_gauss"] | \
            wake.wake_deflection_parameters["empirical_gauss"] | \
            wake.wake_turbulence_parameters["wake_induced_mixing"]
        model_parameters["enable_yaw_added_recovery"] = wake.enable_yaw_added_recovery
        model_parameters["enable_active_wake_mixing"] = wake.enable_active_wake_mixing
    elif vel_model == "none":
        model_parameters = {}
    else:
        model_parameters = None

    return model_parameters
