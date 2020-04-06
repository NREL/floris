# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

from ..utilities import Vec3
from ..utilities import wrap_180
from .turbine import Turbine
import numpy as np


class TurbineMap():
    """
    TurbineMap contains instances of Turbine for the wind farm.

    TurbineMap is container object which maps a Turbine instance to a 
    :py:class:`floris.utilities.Vec3` object. This class also provides 
    some helper methods for sorting and manipulating the turbine layout.

    TODO: this docstring is incorrect!

    Args:
        turbine_map_dict: A dictionary mapping of 
            :py:class:`floris.simulation.turbine.Turbine` to 
            :py:class:`floris.utilities.Vec3`; it should have the 
            following form:

            {

                Vec3(): Turbine(),

                Vec3(): Turbine(),

                ...,`

                Vec3(): Turbine()

            }

    Returns:
        TurbineMap: An instantiated TurbineMap object.
    """

    def __init__(self, layout_x, layout_y, turbines):
        """
        all are lists
        """
        coordinates = [Vec3(x1,x2,0 ) for x1, x2 in list(zip(layout_x, layout_y))]
        self._turbine_map_dict = self._build_internal_dict(coordinates, turbines)

    def _build_internal_dict(self, coordinates, turbines):
        turbine_dict = {}
        for i, c in enumerate(coordinates):
            this_coordinate = Vec3(c.x1, c.x2, turbines[i].hub_height)
            turbine_dict[this_coordinate] = turbines[i]
        return turbine_dict

    def update_hub_heights(self):
        """
        If turbine hub-heights have changed make sure to
        update in coordinates
        """
        self._turbine_map_dict = self._build_internal_dict(self.coords, self.turbines)

    def rotated(self, angle, center_of_rotation):
        """
        Rotate the Turbines in TurbineMap by a specific angle.

        Rotate the turbine coordinates by a given angle about a given 
        center of rotation. This function returns a new TurbineMap 
        object whose turbines are rotated. The original TurbineMap is 
        not modified.

        Args:
            angle: A list of the wind direction angles at each turbine,
            in degrees, of which to rotate the turbines.
            center_of_rotation: The center of rotation. If not supplied 
                the default is the center of the flow field.

        Returns:
            TurbineMap: A rotated TurbineMap.
        """
        layout_x = np.zeros(len(self.coords))
        layout_y = np.zeros(len(self.coords))
        for i, coord in enumerate(self.coords):
            coord.rotate_on_x3(angle[i], center_of_rotation)
            layout_x[i] = coord.x1prime
            layout_y[i] = coord.x2prime
        return TurbineMap(layout_x, layout_y, self.turbines)

    def sorted_in_x_as_list(self):
        """
        Returns a sorted list of turbine coordinates and turbines from 
        smallest x1 coordinate to largest x1 coordinate.

        Returns:
            [Vec3, Turbine]: A sorted list of turbine coordinates and 
            turbines.
        """
        coords = sorted(self._turbine_map_dict, key=lambda coord: coord.x1)
        return [(c, self._turbine_map_dict[c]) for c in coords]

    def number_of_wakes_iec(self, wd, return_turbines=True):
        """
        Returns a dictionary containing the list of turbines in 
        TurbineMap and the total number of wakes from other turbines 
        that interact with each turbine for the provided wind 
        direction. Waked directions are determined using the formula 
        in Figure A.1 in Annex A of the IEC 61400-12-1:2017 standard. 

        Args:
            wd (float): Wind direction for determining waked turbines.

            return_turbines (bool): Should turbines be returned with list

        Returns:
            wake_list: List of Turbine objects and number of upstream 
            turbines waking them.
        """

        wake_list =[]
        for coord0, turbine0 in self.items:

            other_turbines = [(coord, turbine) for coord,turbine in \
                self.items if turbine != turbine0]

            dists = np.array([np.hypot(coord.x1-coord0.x1,coord.x2-coord0.x2)/ \
                turbine.rotor_diameter for coord,turbine in other_turbines])

            angles = np.array([np.degrees(np.arctan2(coord.x1-coord0.x1, \
                coord.x2-coord0.x2)) for coord,turbine in self.items if \
                turbine != turbine0])

            # angles = (-angles - 90) % 360
            
            waked = dists <= 2.
            waked = waked | ((dists <= 20.) & (np.abs(wrap_180(wd-angles)) \
                <= 0.5*(1.3*np.degrees(np.arctan(2.5/dists+0.15))+10)))

            if return_turbines:
                wake_list.append((turbine0,waked.sum()))
            else:
                wake_list.append(waked.sum())
        
        return wake_list

    @property
    def turbines(self):
        """
        Property that returns list of Turbine objects in wind farm.

        Returns:
            [Turbine]: A list of Turbine objects.
        """
        return [turbine for _, turbine in self.items]

    @property
    def coords(self):
        """
        Property that returns a list of coordinates of the turbines in 
        a wind farm.

        Returns:
            [Vec3]: A list of turbine coordinates.
        """
        return [coord for coord, _ in self.items]

    @property
    def items(self):
        """
        Property that returns a list with dictionary pairs of 
        coordinates and Turbine objects.

        Returns:
            dict_items: List of dictionary key and value pairs of 
            coordinates and Turbine objects.
        """
        return self._turbine_map_dict.items()
