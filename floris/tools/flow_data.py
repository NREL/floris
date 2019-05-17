# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import numpy as np
from ..utilities import Vec3, Output


class FlowData():
    """
    Generate a FlowData object to handle data I/O
    """

    #TODO handle none case, maybe defaul values apply like 0 origin and auto determine spacing and dimensions
    def __init__(self,
                 x,
                 y,
                 z,
                 u,
                 v,
                 w,
                 spacing=None,
                 dimensions=None,
                 origin=None):
        """
        Initialize FlowData object with coordinates, velocity fields,
        and meta data.

        Args:
            x (np.array): Cartesian coordinate data.
            y (np.array): Cartesian coordinate data.
            z (np.array): Cartesian coordinate data.
            u (np.array): x-component of velocity.
            v (np.array): y-component of velocity.
            w (np.array): z-component of velocity.
            spacing (float, optional): Spatial resolution.
                Defaults to None.
            dimensions (iterable, optional): named dimensions
                (e.g. x1, x2, x3). Defaults to None.
            origin (iterable, optional): Coordinates of origin.
                Defaults to None.
        """

        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v
        self.w = w

        #TODO Make these VEC3?
        self.spacing = spacing
        self.dimensions = dimensions
        self.origin = origin

        # Technically resolution is a restating of above, but it is useful to have
        self.resolution = Vec3(len(np.unique(x)), len(np.unique(y)),
                               len(np.unique(z)))

    def save_as_vtk(self, filename):
        """
        Save FlowData Object to vtk

        Args:
            filename (str): Write-to path for vtk file
        """
        n_points = self.dimensions.x1 * self.dimensions.x2 * self.dimensions.x3
        vtk_file = Output(filename)
        vtk_file.write_line('# vtk DataFile Version 3.0')
        vtk_file.write_line('array.mean0D')
        vtk_file.write_line('ASCII')
        vtk_file.write_line('DATASET STRUCTURED_POINTS')
        vtk_file.write_line('DIMENSIONS {}'.format(self.dimensions))
        vtk_file.write_line('ORIGIN {}'.format(self.origin))
        vtk_file.write_line('SPACING {}'.format(self.spacing))
        vtk_file.write_line('POINT_DATA {}'.format(n_points))
        vtk_file.write_line('FIELD attributes 1')
        vtk_file.write_line('UAvg 3 {} float'.format(n_points))
        for u, v, w in zip(self.u, self.v, self.w):
            vtk_file.write_line('{}'.format(Vec3(u, v, w)))

    @staticmethod
    def crop(ff, x_bnds, y_bnds, z_bnds):
        """
        Crop FlowData object to within stated bounds.

        Args:
            ff (:py:class:`floris.tools.flow_data.FlowData`):
                FlowData object.
            x_bnds (iterable): min and max of x-coordinate.
            y_bnds (iterable): min and max of y-coordinate.
            z_bnds (iterable): min and max of z-coordinate.

        Returns:
            (:py:class:`floris.tools.flow_data.FlowData`):
            cropped FlowData object.
        """

        map_values = (ff.x > x_bnds[0]) & (ff.x < x_bnds[1]) & (
            ff.y > y_bnds[0]) & (ff.y < y_bnds[1]) & (ff.z > z_bnds[0]) & (
                ff.z < z_bnds[1])

        x = ff.x[map_values]
        y = ff.y[map_values]
        z = ff.z[map_values]

        
        dimensions = Vec3(len(np.unique(x)), len(np.unique(y)), len(np.unique(z)))

        # Work out origin
        origin = Vec3(
            ff.origin.x1 + np.min(x),
            ff.origin.x2 + np.min(y),
            ff.origin.x3 + np.min(z),
        )


        return FlowData(
            x - np.min(x),
            y - np.min(y),
            z - np.min(z),
            ff.u[map_values],
            ff.v[map_values],
            ff.w[map_values],
            spacing=ff.spacing,  # doesn't change
            dimensions=dimensions,
            origin=origin)
