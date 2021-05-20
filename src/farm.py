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
from .turbine import Turbine
from .utilities import Vec3, wrap_180
from .logging_manager import LoggerBase
from typing import Dict

class Farm:
    """
    Farm is a class containing the objects that make up a FLORIS model.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.
    """

    def __init__(self, input_dictionary: Dict, turbine: Turbine):
        """
        The initialization method unpacks some of the data from the input
        dictionary in order to create a couple of unerlying data structures:

            - :py:obj:`~.wind_map.WindMap`

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
        wind_x = input_dictionary["wind_x"]
        wind_y = input_dictionary["wind_y"]

        # check if the length of x and y coordinates are equal
        if len(layout_x) != len(layout_y):
            err_msg = (
                "The number of turbine x locations ({0}) is "
                + "not equal to the number of turbine y locations "
                + "({1}). Please check your layout array."
            ).format(len(layout_x), len(layout_y))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        coordinates = [Vec3([x1, x2, 0]) for x1, x2 in list(zip(layout_x, layout_y))]
        self.turbine_map_dict = self._build_internal_dict(coordinates, [copy.deepcopy(turbine) for ii in range(len(layout_x))])
        self.wind_map = WindMap(
            wind_speed=input_dictionary["wind_speed"],
            layout_array=(layout_x, layout_y),
            wind_layout=(wind_x, wind_y),
            turbulence_intensity=input_dictionary["turbulence_intensity"],
            wind_direction=input_dictionary["wind_direction"],
        )

    def _build_internal_dict(self, coordinates, turbines):
        turbine_dict = {}
        for i, c in enumerate(coordinates):
            this_coordinate = Vec3([c.x1, c.x2, turbines[i].hub_height])
            turbine_dict[this_coordinate] = turbines[i]
        return turbine_dict

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

