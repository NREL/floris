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


import os

import numpy as np
from sklearn import neighbors

from ..utilities import Vec3


class FlowData:
    """
    FlowData objects represent a saved 3D flow from a FLORIS simulation
    or other data source.
    """

    # TODO handle none case, maybe default values apply like 0 origin and auto
    # determine spacing and dimensions
    def __init__(self, x, y, z, u, v, w, spacing=None, dimensions=None, origin=None):
        """
        Initialize FlowData object with coordinates, velocity fields,
        and meta-data.

        Args:
            x (np.array): Cartesian coordinate data.
            y (np.array): Cartesian coordinate data.
            z (np.array): Cartesian coordinate data.
            u (np.array): x-component of velocity.
            v (np.array): y-component of velocity.
            w (np.array): z-component of velocity.
            spacing (float, optional): Spatial resolution.
                Defaults to None.
            dimensions (iterable, optional): Named dimensions
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

        # TODO Make these VEC3?
        self.spacing = spacing
        self.dimensions = dimensions
        self.origin = origin

        # Technically resolution is a restating of above, but it is useful to have
        self.resolution = Vec3(len(np.unique(x)), len(np.unique(y)), len(np.unique(z)))

    def save_as_vtk(self, filename):
        """
        Save FlowData Object to vtk format.

        Args:
            filename (str): Write-to path for vtk file.
        """
        n_points = self.dimensions.x1 * self.dimensions.x2 * self.dimensions.x3

        ln = "\n"
        vtk_file = open(filename, "w")
        vtk_file.write("# vtk DataFile Version 3.0" + ln)
        vtk_file.write("array.mean0D" + ln)
        vtk_file.write("ASCII" + ln)
        vtk_file.write("DATASET STRUCTURED_POINTS" + ln)
        vtk_file.write("DIMENSIONS {}".format(self.dimensions) + ln)
        vtk_file.write("ORIGIN {}".format(self.origin) + ln)
        vtk_file.write("SPACING {}".format(self.spacing) + ln)
        vtk_file.write("POINT_DATA {}".format(n_points) + ln)
        vtk_file.write("FIELD attributes 1" + ln)
        vtk_file.write("UAvg 3 {} float".format(n_points) + ln)
        for u, v, w in zip(self.u, self.v, self.w):
            vtk_file.write_line("{}".format(Vec3(u, v, w)) + ln)

    @staticmethod
    def crop(ff, x_bnds, y_bnds, z_bnds):
        """
        Crop FlowData object to within stated bounds.

        Args:
            ff (:py:class:`~.tools.flow_data.FlowData`):
                FlowData object.
            x_bnds (iterable): Min and max of x-coordinate.
            y_bnds (iterable): Min and max of y-coordinate.
            z_bnds (iterable): Min and max of z-coordinate.

        Returns:
            (:py:class:`~.tools.flow_data.FlowData`):
            Cropped FlowData object.
        """

        map_values = (
            (ff.x > x_bnds[0])
            & (ff.x < x_bnds[1])
            & (ff.y > y_bnds[0])
            & (ff.y < y_bnds[1])
            & (ff.z > z_bnds[0])
            & (ff.z < z_bnds[1])
        )

        x = ff.x[map_values]
        y = ff.y[map_values]
        z = ff.z[map_values]

        #  Work out new dimensions
        dimensions = Vec3(len(np.unique(x)), len(np.unique(y)), len(np.unique(z)))

        # Work out origin
        origin = (
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
            origin=origin,
        )

        # Define a quick function for getting arbitrary points from sowfa

    def get_points_from_flow_data(self, x_points, y_points, z_points):
        """
        Return the u-value of a set of points from with a FlowData object.
        Use a simple nearest neighbor regressor to do internal interpolation.

        Args:
            x_points (np.array): Array of x-locations of points.
            y_points (np.array): Array of y-locations of points.
            z_points (np.array): Array of z-locations of points.

        Returns:
            np.array: Array of u-velocity at specified points.
        """
        # print(x_points,y_points,z_points)
        X = np.column_stack([self.x, self.y, self.z])
        n_neighbors = 1
        knn = neighbors.KNeighborsRegressor(n_neighbors)
        y_ = knn.fit(X, self.u)  # .predict(T)

        # Predict new points
        T = np.column_stack([x_points, y_points, z_points])
        return knn.predict(T)
