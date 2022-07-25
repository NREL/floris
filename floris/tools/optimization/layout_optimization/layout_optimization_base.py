# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

from ....logging_manager import LoggerBase

class LayoutOptimization(LoggerBase):
    def __init__(self, fi, boundaries, min_dist=None, freq=None):
        self.fi = fi.copy()
        self.boundaries = boundaries

        self._boundary_polygon = Polygon(self.boundaries)
        self._boundary_line = LineString(self.boundaries)

        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])
        self.x0 = self._norm(self.fi.layout_x, self.xmin, self.xmax)
        self.y0 = self._norm(self.fi.layout_y, self.ymin, self.ymax)

        if min_dist is None:
            self.min_dist = 2 * self.rotor_diameter
        else:
            self.min_dist = min_dist

        if freq is None:
            self.freq = 1
        else:
            self.freq = freq

        self.wdir = self.fi.floris.flow_field.wind_directions
        self.wspd = self.fi.floris.flow_field.wind_speeds
        self.initial_AEP = np.sum(self.fi.get_farm_power() * self.freq * 8760)

    def __str__(self):
        return "layout"

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    # Public methods required for each optimization class

    def optimize(self):
        sol = self._optimize()
        return sol

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = self.fi.floris.farm.n_turbines
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fi.floris.farm.rotor_diameters[0][0][0]