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


import pickle
import json

import src.logging_manager as logging_manager

from .farm import Farm
from .wake import Wake
from .turbine import Turbine


class Floris(logging_manager.LoggerBase):
    """
    Top-level class that describes a Floris model and initializes the
    simulation. Use the :py:class:`~.simulation.farm.Farm` attribute to
    access other objects within the model.
    """

    def __init__(self, input_file=None, input_dict=None):
        """
        Import this class with one of the two optional input arguments
        to create a Floris model. The `input_dict` and `input_file`
        should both conform to the same data format.

        Args:
            input_file (str, optional): Path to the input file which will
                be parsed and converted to a Python dict.
            input_dict (dict, optional): Python dict given directly.
        """
        # Parse the input into dictionaries
        if input_file is not None:
            with open(input_file) as jsonfile:
                input_dict = json.load(jsonfile)
            input_dict = self._parseJSON(input_file)
        elif input_dict is not None:
            input_dict = input_dict.copy()
        else:
            raise ValueError("InputReader: No input file or dictionary given.")

        turbine_dict = input_dict.pop("turbine")
        wake_dict = input_dict.pop("wake")
        farm_dict = input_dict.pop("farm")
        meta_dict = input_dict

        # Configure logging
        logging_manager.configure_console_log(
            meta_dict["logging"]["console"]["enable"],
            meta_dict["logging"]["console"]["level"],
        )
        logging_manager.configure_file_log(
            meta_dict["logging"]["file"]["enable"],
            meta_dict["logging"]["file"]["level"],
        )

        # Initialize the simulation objects
        self.turbine = Turbine(turbine_dict)
        self.wake = Wake(wake_dict)
        self.farm = Farm(farm_dict)

    def set_wake_model(self, wake_model):
        """
        Sets the velocity deficit model to use as given, and determines the
        wake deflection model based on the selected velocity deficit model.

        Args:
            wake_model (str): The desired wake model.

        Raises:
            Exception: Invalid wake model.
        """
        valid_wake_models = [
            "jensen",
            "turbopark",
            "multizone",
            "gauss",
            "gauss_legacy",
            "blondel",
            "ishihara_qian",
            "curl",
        ]
        if wake_model not in valid_wake_models:
            # TODO: logging
            raise Exception(
                "Invalid wake model. Valid options include: {}.".format(
                    ", ".join(valid_wake_models)
                )
            )

        self.flow_field.wake.velocity_model = wake_model
        if (
            wake_model == "jensen"
            or wake_model == "multizone"
            or wake_model == "turbopark"
        ):
            self.flow_field.wake.deflection_model = "jimenez"
        elif (
            wake_model == "blondel"
            or wake_model == "ishihara_qian"
            or "gauss" in wake_model
        ):
            self.flow_field.wake.deflection_model = "gauss"
        else:
            self.flow_field.wake.deflection_model = wake_model

        self.flow_field.reinitialize_flow_field(
            with_resolution=self.flow_field.wake.velocity_model.model_grid_resolution
        )

        self.reinitialize_turbines()

    def set_yaw_angles(self, yaw_angles):
        """
        Sets the yaw angles for all turbines on the
        :py:obj:`~.turbine.Turbine` objects directly.

        Args:
            yaw_angles (float or list( float )): A single value to set
                all turbine yaw angles or a list of yaw angles corresponding
                to individual turbine yaw angles. Yaw angles are expected
                in degrees.
        """
        if isinstance(yaw_angles, float) or isinstance(yaw_angles, int):
            yaw_angles = [yaw_angles] * len(self.turbines)

        for yaw_angle, turbine in zip(yaw_angles, self.turbines):
            turbine.yaw_angle = yaw_angle

    def update_hub_heights(self):
        """
        Triggers a rebuild of the internal Python dictionary. This may be
        used to update the z-component of the turbine coordinates if
        the hub height has changed.
        """
        self.turbine_map_dict = self._build_internal_dict(self.coords, self.turbines)

    def number_of_wakes_iec(self, wd, return_turbines=True):
        """
        Finds the number of turbines waking each turbine for the given
        wind direction. Waked directions are determined using the formula
        in Figure A.1 in Annex A of the IEC 61400-12-1:2017 standard.
        # TODO: Add the IEC standard as a reference.

        Args:
            wd (float): Wind direction for determining waked turbines.
            return_turbines (bool, optional): Switch to return turbines.
                Defaults to True.

        Returns:
            list(int) or list( (:py:class:`~.turbine.Turbine`, int ) ):
            Number of turbines waking each turbine and, optionally,
            the list of Turbine objects in the map.

        TODO:
        - This could be reworked so that the return type is more consistent.
        - Describe the method used to find upstream turbines.
        """
        wake_list = []
        for coord0, turbine0 in self.items:

            other_turbines = [
                (coord, turbine) for coord, turbine in self.items if turbine != turbine0
            ]

            dists = np.array(
                [
                    np.hypot(coord.x1 - coord0.x1, coord.x2 - coord0.x2)
                    / turbine.rotor_diameter
                    for coord, turbine in other_turbines
                ]
            )

            angles = np.array(
                [
                    np.degrees(np.arctan2(coord.x1 - coord0.x1, coord.x2 - coord0.x2))
                    for coord, turbine in self.items
                    if turbine != turbine0
                ]
            )

            # angles = (-angles - 90) % 360

            waked = dists <= 2.0
            waked = waked | (
                (dists <= 20.0)
                & (
                    np.abs(wrap_180(wd - angles))
                    <= 0.5 * (1.3 * np.degrees(np.arctan(2.5 / dists + 0.15)) + 10)
                )
            )

            if return_turbines:
                wake_list.append((turbine0, waked.sum()))
            else:
                wake_list.append(waked.sum())

        return wake_list