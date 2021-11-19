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

import json
import pickle
from pathlib import Path

import attr
import yaml

import src.logging_manager as logging_manager
from src.utilities import FromDictMixin
from src.simulation import (
    Farm,
    Wake,
    Turbine,
    FlowField,
    TurbineGrid,
    sequential_solver,
)
from src.simulation.wake_velocity import CurlVelocityDeficit, JensenVelocityDeficit
from src.simulation.wake_deflection import JimenezVelocityDeflection


MODEL_MAP = {
    # "wake_combination": {"""The Combination Models"""},
    "wake_deflection": {"jimenez": JimenezVelocityDeflection},
    # "wake_turbulence": {"""The Turbulence Models"""},
    "wake_velocity": {"curl": CurlVelocityDeficit, "jensen": JensenVelocityDeficit},
}
VALID_WAKE_MODELS = [
    # NOTE: These are all models I've applied the attrs routines to
    # "blondel",
    "curl",
    # "gauss",
    # "gauss_legacy",
    "ishihara_qian",
    "jensen",
    "jimenez"
    # "multizone",
    # "turbopark",
]


def convert_dict_to_turbine(turbine_map: dict[str, dict]) -> dict[str, Turbine]:
    """Converts the dictionary of turbine input data to a dictionary of `Turbine`s.

    Args:
        turbine_map (dict[str, dict]): The "turbine" dictionary from the input file/dictionary.

    Returns:
        dict[str, Turbine]: The dictionary of `Turbine`s.
    """
    return {key: Turbine.from_dict(val) for key, val in turbine_map.items()}


@attr.s(auto_attribs=True)
class Floris(logging_manager.LoggerBase, FromDictMixin):
    """
    Top-level class that describes a Floris model and initializes the
    simulation. Use the :py:class:`~.simulation.farm.Farm` attribute to
    access other objects within the model.
    """

    farm: Farm = attr.ib()
    logging: dict = attr.ib()
    turbine: dict[str, Turbine] = attr.ib(converter=convert_dict_to_turbine)
    wake: Wake = attr.ib(converter=Wake.from_dict)
    flow_field: FlowField = attr.ib(converter=FlowField.from_dict)

    def __attrs_post_init__(self) -> None:
        self.create_farm()

        # Configure logging
        logging_manager.configure_console_log(
            self.logging["console"]["enable"],
            self.logging["console"]["level"],
        )
        logging_manager.configure_file_log(
            self.logging["file"]["enable"],
            self.logging["file"]["level"],
        )

    @classmethod
    def from_json(cls, input_file_path: str | Path) -> Floris:
        """Creates a `Floris` instance from a JSON file.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                JSON input file.

        Returns:
            Floris: The class object instance.
        """
        input_file_path = Path(input_file_path).resolve()
        with open(input_file_path) as json_file:
            input_dict = json.load(json_file)
        return Floris.from_dict(input_dict)

    @classmethod
    def from_yaml(cls, input_file_path: str | Path) -> Floris:
        """Creates a `Floris` instance from a YAML file.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                YAML input file.

        Returns:
            Floris: The class object instance
        """
        input_file_path = Path(input_file_path).resolve()
        input_dict = yaml.load(open(input_file_path, "r"), Loader=yaml.SafeLoader)
        return Floris.from_dict(input_dict)

    def _prepare_for_save(self) -> dict:
        output_dict = dict(
            farm=attr.asdict(self.farm),
            logging=self.logging,
            turbine={key: attr.asdict(val) for key, val in self.turbine.items()},
            wake=attr.asdict(self.wake),
            flow_field=attr.asdict(self.flow_field),
        )
        return output_dict

    def to_json(self, output_file_path: str) -> None:
        """Converts the `Floris` object to an input-ready JSON file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the JSON file.
        """
        output_dict = self._prepare_for_save()
        with open(output_file_path, "w+") as f:
            yaml.dump(output_dict, f, indent=2, sort_keys=False)

    def to_yaml(self, output_file_path: str) -> None:
        """Converts the `Floris` object to an input-ready YAML file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the YAML file.
        """
        output_dict = self._prepare_for_save()
        with open(output_file_path, "w+") as f:
            yaml.dump(output_dict, f, default_flow_style=False)

    def create_farm(self) -> None:
        if len(self.farm["turbine_id"]) == 0:
            self.farm["turbine_id"] = [*self.turbine.keys()][0] * len(self.farm["layout_x"])
        self.farm["turbine_map"] = self.turbine
        self.farm = Farm.from_dict(self.farm)

    def annual_energy_production(self, wind_rose):
        # self.steady_state_atmospheric_condition()
        pass

    def steady_state_atmospheric_condition(self):

        # <<interface>>
        # Initialize grid and field quanitities
        grid = TurbineGrid(
            turbine_coordinates=self.farm.coordinates,
            reference_turbine_diameter=self.farm.reference_turbine_diameter,
            wind_directions=self.flow_field.wind_directions,
            wind_speeds=self.flow_field.wind_speeds,
            grid_resolution=5,
        )
        # TODO: where do we pass in grid_resolution? Hardcoded to 5 above.

        self.flow_field.initialize_velocity_field(grid)

        # <<interface>>
        # JensenVelocityDeficit.solver(self.farm, self.flow_field)
        sequential_solver(self.farm, self.flow_field, grid)

        grid.finalize()
        self.flow_field.finalize(grid.unsorted_indices)

    # Utility functions

    def update_hub_heights(self):
        """
        Triggers a rebuild of the internal Python dictionary. This may be
        used to update the z-component of the turbine coordinates if
        the hub height has changed.
        """
        self.turbine_map_dict = self._build_internal_dict(self.coords, self.turbines)
