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

from .wind_map import WindMap
from .utilities import Vec3
from .flow_field import FlowField


class Farm:
    """
    Farm is a class containing the objects that make up a FLORIS model.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.
    """

    def __init__(self, instance_dictionary, turbine, wake):
        """
        The initialization method unpacks some of the data from the input
        dictionary in order to create a couple of unerlying data structures:

            - :py:obj:`~.wind_map.WindMap`

        Args:
            instance_dictionary (dict): The required keys in this dictionary
                are:

                    -   **wind_speed** (*list*): The wind speed measurements at
                        hub height (m/s).
                    -   **wind_x** (*list*): The x-coordinates of the wind
                        speed measurements.
                    -   **wind_y** (*list*): The y-coordinates of the wind
                        speed measurements.
                    -   **wind_direction** (*list*): The wind direction
                        measurements (deg).
                    -   **turbulence_intensity** (*list*): Turbulence intensity
                        measurements at hub height (as a decimal fraction).
                    -   **wind_shear** (*float*): The power law wind shear
                        exponent.
                    -   **wind_veer** (*float*): The vertical change in wind
                        direction across the rotor.
                    -   **air_density** (*float*): The air density (kg/m^3).
                    -   **layout_x** (*list*): The x-coordinates of the
                        turbines.
                    -   **layout_y** (*list*): The y-coordinates of the
                        turbines.

            turbine (:py:obj:`~.turbine.Turbine`): The turbine models used
                throughout the farm.
            wake (:py:obj:`~.wake.Wake`): The wake model used to simulate the
                freestream flow and wakes.
        """
        self.name = instance_dictionary["name"]
        properties = instance_dictionary["properties"]
        layout_x = properties["layout_x"]
        layout_y = properties["layout_y"]
        wind_x = properties["wind_x"]
        wind_y = properties["wind_y"]

        self.wind_map = WindMap(
            wind_speed=properties["wind_speed"],
            layout_array=(layout_x, layout_y),
            wind_layout=(wind_x, wind_y),
            turbulence_intensity=properties["turbulence_intensity"],
            wind_direction=properties["wind_direction"],
        )

        self.flow_field = FlowField(
            wind_shear=properties["wind_shear"],
            wind_veer=properties["wind_veer"],
            air_density=properties["air_density"],
            ),
            wake=wake,
            wind_map=self.wind_map,
            specified_wind_height=properties["specified_wind_height"],
        )

    def __str__(self):
        return (
            "Name: {}\n".format(self.name)
            + "Wake Model: {}\n".format(self.flow_field.wake.velocity_model)
            + "Deflection Model: {}\n".format(self.flow_field.wake.deflection_model)
        )

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

    @property
    def wind_direction(self):
        """
        Wind direction at each wind turbine.
        # TODO: Explain the wind direction change here.
        #       - Is there a transformation on wind map?
        #       - Is this always from a particular direction?

        Returns:
            list(float)
        """
        return list((np.array(self.wind_map.turbine_wind_direction) - 90) % 360)

    def update_hub_heights(self):
        """
        Triggers a rebuild of the internal Python dictionary. This may be
        used to update the z-component of the turbine coordinates if
        the hub height has changed.
        """
        self.turbine_map_dict = self._build_internal_dict(self.coords, self.turbines)

    def rotated(self, angles, center_of_rotation):
        """
        Rotates each turbine coordinate by a given angle about a center
        of rotation.

        Args:
            angles ( list(float) ): Angles in degrees to rotate each turbine.
            center_of_rotation ( :py:class:`~.utilities.Vec3` ):
                The center of rotation.

        Returns:
            :py:class:`~.turbine_map.TurbineMap`: A new TurbineMap object whose
            turbines are rotated from the original.
        """
        layout_x = np.zeros(len(self.coords))
        layout_y = np.zeros(len(self.coords))
        for i, coord in enumerate(self.coords):
            coord.rotate_on_x3(angles[i], center_of_rotation)
            layout_x[i] = coord.x1prime
            layout_y[i] = coord.x2prime
        return TurbineMap(layout_x, layout_y, self.turbines)

    def sorted_in_x_as_list(self):
        """
        Sorts the turbines based on their x-coordinates in ascending order.

        Returns:
            list((:py:class:`~.utilities.Vec3`, :py:class:`~.turbine.Turbine`)):
            The sorted coordinates and corresponding turbines. This is a
            list of tuples where each tuple contains the coordinate
            and turbine in the first and last element, respectively.
        """
        coords = sorted(self.turbine_map_dict, key=lambda coord: coord.x1)
        return [(c, self.turbine_map_dict[c]) for c in coords]

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

    def reinitialize_turbines(self, air_density=None):
        for turbine in self.turbines:
            turbine.reinitialize(air_density)

    @property
    def turbines(self):
        """
        Turbines contained in the :py:class:`~.turbine_map.TurbineMap`.

        Returns:
            list(:py:class:`floris.simulation.turbine.Turbine`)
        """
        return [turbine for _, turbine in self.items]

    @property
    def wake(self):
        """
        The Farm's Wake object. This is used to reduce the depth of the
        object-hierachy required to modify the wake models from a script.

        Returns:
            :py:obj:`~.wake.Wake`.
        """
        return self.flow_field.wake

    @property
    def coords(self):
        """
        Coordinates of the turbines contained in the
        :py:class:`~.turbine_map.TurbineMap`.

        Returns:
            list(:py:class:`~.utilities.Vec3`)
        """
        return [coord for coord, _ in self.items]

    @property
    def items(self):
        """
        Contents of the internal Python dictionary mapping of the turbine
        and coordinates.

        Returns:
            dict_items: Iterable object containing tuples of key-value pairs
            where the first index is the coordinate
            (:py:class:`~.utilities.Vec3`) and the second index is the
            :py:class:`~.turbine.Turbine`.
        """
        return self.turbine_map_dict.items()

