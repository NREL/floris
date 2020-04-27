# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation
 

from ..utilities import Vec3
from ..utilities import wrap_180
from .turbine import Turbine
import numpy as np


class TurbineMap():
    """
    Container object that maps a :py:class:`~.turbine.Turbine` instance to a
    :py:class:`~.utilities.Vec3` object. This class also provides some helper
    methods for sorting and manipulating the turbine layout.

    The underlying data structure for this class is a Python dictionary
    in the following form:

        {

            Vec3(): Turbine(),

            Vec3(): Turbine(),

            ...,

        }
    """

    def __init__(self, layout_x, layout_y, turbines):
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
        Triggers a rebuild of the internal Python dictionary. This may be
        used to update the z-component of the turbine coordinates if
        the hub height has changed.
        """
        self._turbine_map_dict = self._build_internal_dict(self.coords, self.turbines)

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
        coords = sorted(self._turbine_map_dict, key=lambda coord: coord.x1)
        return [(c, self._turbine_map_dict[c]) for c in coords]

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
        return self._turbine_map_dict.items()
