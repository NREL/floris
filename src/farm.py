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
from .utilities import Vec3
from .turbine import Turbine


class FarmController:
    def __init__(self, n_wind_speeds: int, n_wind_directions: int) -> None:
        self.yaw_angles = []
    
    def set_yaw_angles(self, yaw_angles: list) -> None:
        self.yaw_angles = yaw_angles


class Farm:
    """
    Farm is a class containing the objects that make up a FLORIS model.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.
    """

    def __init__(self, input_dictionary: dict, turbine: Turbine):
        """
        Args:
            input_dictionary (dict): The required keys in this dictionary
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
        """
        Converts input coordinates into :py:class:`~.utilities.Vec3` and
        constructs the underlying mapping to :py:class:`~.turbine.Turbine`.
        It is assumed that all arguments are of the same length and that the
        Turbine at a particular index corresponds to the coordinate at the same
        index in the layout arguments.

        Args:
            layout_x ( list(float) ): X-coordinate of the turbine locations.
            layout_y ( list(float) ): Y-coordinate of the turbine locations.
            turbines ( list(float) ): Turbine objects corresponding to
                the locations given in layout_x and layout_y.
        """
        layout_x = input_dictionary["layout_x"]
        layout_y = input_dictionary["layout_y"]

        # check if the length of x and y coordinates are equal
        if len(layout_x) != len(layout_y):
            err_msg = (
                "The number of turbine x locations ({0}) is "
                + "not equal to the number of turbine y locations "
                + "({1}). Please check your layout array."
            ).format(len(layout_x), len(layout_y))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        coordinates = [Vec3([x1, x2, turbine.hub_height]) for x1, x2 in list(zip(layout_x, layout_y))]        
        self.turbine_map_dict = {c: copy.deepcopy(turbine) for c in coordinates}

        # Turbine control settings indexed by the turbine ID
        self.farm_controller = FarmController(len(input_dictionary["wind_speeds"]), 1)
        self.farm_controller.set_yaw_angles([0] * len(self.turbine_map_dict))

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

    @property
    def turbines(self):
        """
        Turbines contained in the :py:class:`~.turbine_map.TurbineMap`.

        Returns:
            list(:py:class:`floris.simulation.turbine.Turbine`)
        """
        return [turbine for _, turbine in self.items]

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

    def set_yaw_angles(self, yaw_angles: list, n_wind_speeds: int, n_wind_directions: int) -> None:
        if len(yaw_angles) != len(self.items):
            raise ValueError("Farm.set_yaw_angles: a yaw angle must be given for each turbine.")
        # TODO: support a user-given yaw angle setting for each wind speed and wind direction

        self.farm_controller.set_yaw_angles(
            np.reshape(
                np.array([yaw_angles] * n_wind_speeds),  # broadcast
                (len(self.items), n_wind_speeds)  # reshape
            )
        )
