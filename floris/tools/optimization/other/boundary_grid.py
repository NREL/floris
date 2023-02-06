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


import numpy as np
from shapely.geometry import Point, Polygon


def discontinuous_grid(
    nrows,
    ncols,
    farm_width,
    farm_height,
    shear,
    rotation,
    center_x,
    center_y,
    shrink_boundary,
    boundary_x,
    boundary_y,
    eps=1e-3,
):
    """
    Map from grid design variables to turbine x and y locations.
    Includes integer design variables and the formulation
    results in a discontinous design space.

    TODO: shrink_boundary doesn't work well with concave boundaries,
    or with boundary angles less than 90 deg

    Args:
        nrows (Int): number of rows in the grid.
        ncols (Int): number of columns in the grid.
        farm_width (Float): total grid width (before shear).
        farm_height (Float): total grid height.
        shear (Float): grid shear (rad).
        rotation (Float): rotation about grid center (rad).
        center_x (Float): location of grid x center.
        center_y (Float): location of grid y center.
        shrink_boundary (Float): how much to shrink the boundary that the grid can occupy.
        boundary_x (Array(Float)): x boundary points.
        boundary_y (Array(Float)): y boundary points.

    Returns:
        grid_x (Array(Float)): turbine x locations.
        grid_y (Array(Float)): turbine y locations.
    """
    # create grid
    nrows = int(nrows)
    ncols = int(ncols)
    xlocs = np.linspace(0.0, farm_width, ncols)
    ylocs = np.linspace(0.0, farm_height, nrows)
    y_spacing = ylocs[1] - ylocs[0]
    nturbs = nrows * ncols
    grid_x = np.zeros(nturbs)
    grid_y = np.zeros(nturbs)
    turb = 0
    for i in range(nrows):
        for j in range(ncols):
            grid_x[turb] = xlocs[j] + float(i) * y_spacing * np.tan(shear)
            grid_y[turb] = ylocs[i]
            turb += 1

    # rotate
    grid_x, grid_y = (
        np.cos(rotation) * grid_x - np.sin(rotation) * grid_y,
        np.sin(rotation) * grid_x + np.cos(rotation) * grid_y,
    )

    # move center of grid
    grid_x = (grid_x - np.mean(grid_x)) + center_x
    grid_y = (grid_y - np.mean(grid_y)) + center_y

    # arrange the boundary

    # boundary = np.zeros((len(boundary_x),2))
    # boundary[:,0] = boundary_x[:]
    # boundary[:,1] = boundary_y[:]
    # poly = Polygon(boundary)
    # centroid = poly.centroid

    # boundary[:,0] = (boundary_x[:]-centroid.x)*boundary_mult + centroid.x
    # boundary[:,1] = (boundary_y[:]-centroid.y)*boundary_mult + centroid.y
    # poly = Polygon(boundary)

    boundary = np.zeros((len(boundary_x), 2))
    boundary[:, 0] = boundary_x[:]
    boundary[:, 1] = boundary_y[:]
    poly = Polygon(boundary)

    if shrink_boundary != 0.0:
        nBounds = len(boundary_x)
        for i in range(nBounds):
            point = Point(boundary_x[i] + eps, boundary_y[i])
            if poly.contains(point) is True or poly.touches(point) is True:
                boundary[i, 0] = boundary_x[i] + shrink_boundary
            else:
                boundary[i, 0] = boundary_x[i] - shrink_boundary

            point = Point(boundary_x[i], boundary_y[i] + eps)
            if poly.contains(point) is True or poly.touches(point) is True:
                boundary[i, 1] = boundary_y[i] + shrink_boundary
            else:
                boundary[i, 1] = boundary_y[i] - shrink_boundary

        poly = Polygon(boundary)

    # get rid of points outside of boundary
    index = 0
    for i in range(len(grid_x)):
        point = Point(grid_x[index], grid_y[index])
        if poly.contains(point) is False and poly.touches(point) is False:
            grid_x = np.delete(grid_x, index)
            grid_y = np.delete(grid_y, index)
        else:
            index += 1

    return grid_x, grid_y


def place_boundary_turbines(n_boundary_turbs, start, boundary_x, boundary_y):
    """
    Place turbines equally spaced traversing the perimiter if the wind farm along the boundary

    Args:
    n_boundary_turbs (Int): number of turbines to be placed on the boundary
    start (Float): where the first turbine should be placed
    boundary_x (Array(Float)): x boundary points
    boundary_y (Array(Float)): y boundary points

    Returns
    layout_x (Array(Float)): turbine x locations
    layout_y (Array(Float)): turbine y locations
    """

    # check if the boundary is closed, correct if not
    if boundary_x[-1] != boundary_x[0] or boundary_y[-1] != boundary_y[0]:
        boundary_x = np.append(boundary_x, boundary_x[0])
        boundary_y = np.append(boundary_y, boundary_y[0])

    # make the boundary
    boundary = np.zeros((len(boundary_x), 2))
    boundary[:, 0] = boundary_x[:]
    boundary[:, 1] = boundary_y[:]
    poly = Polygon(boundary)
    perimeter = poly.length

    # get the flattened turbine locations
    spacing = perimeter / float(n_boundary_turbs)
    flattened_locs = np.linspace(start, perimeter + start - spacing, n_boundary_turbs)

    # set all of the flattened values between 0 and the perimeter
    for i in range(n_boundary_turbs):
        while flattened_locs[i] < 0.0:
            flattened_locs[i] += perimeter
        if flattened_locs[i] > perimeter:
            flattened_locs[i] = flattened_locs[i] % perimeter

    # place the turbines around the perimeter
    nBounds = len(boundary_x)
    layout_x = np.zeros(n_boundary_turbs)
    layout_y = np.zeros(n_boundary_turbs)

    lenBound = np.zeros(nBounds - 1)
    for i in range(nBounds - 1):
        lenBound[i] = Point(boundary[i]).distance(Point(boundary[i + 1]))
    for i in range(n_boundary_turbs):
        for j in range(nBounds - 1):
            if flattened_locs[i] < sum(lenBound[0 : j + 1]):
                layout_x[i] = (
                    boundary_x[j]
                    + (boundary_x[j + 1] - boundary_x[j])
                    * (flattened_locs[i] - sum(lenBound[0:j]))
                    / lenBound[j]
                )
                layout_y[i] = (
                    boundary_y[j]
                    + (boundary_y[j + 1] - boundary_y[j])
                    * (flattened_locs[i] - sum(lenBound[0:j]))
                    / lenBound[j]
                )
                break

    return layout_x, layout_y


def boundary_grid(
    n_boundary_turbs,
    start,
    nrows,
    ncols,
    farm_width,
    farm_height,
    shear,
    rotation,
    center_x,
    center_y,
    shrink_boundary,
    boundary_x,
    boundary_y,
    eps=1e-3,
):
    """
    Place turbines equally spaced traversing the perimiter if the wind farm along the boundary

    Args:
    n_boundary_turbs,start: boundary variables
    nrows,ncols,farm_width,farm_height,shear,
        rotation,center_x,center_y,shrink_boundary,eps: grid variables
    boundary_x,boundary_y: boundary points

    Returns
    layout_x (Array(Float)): turbine x locations
    layout_y (Array(Float)): turbine y locations
    """

    boundary_turbines_x, boundary_turbines_y = place_boundary_turbines(
        n_boundary_turbs, start, boundary_x, boundary_y
    )
    grid_turbines_x, grid_turbines_y = discontinuous_grid(
        nrows,
        ncols,
        farm_width,
        farm_height,
        shear,
        rotation,
        center_x,
        center_y,
        shrink_boundary,
        boundary_x,
        boundary_y,
        eps=eps,
    )

    layout_x = np.append(boundary_turbines_x, grid_turbines_x)
    layout_y = np.append(boundary_turbines_y, grid_turbines_y)

    return layout_x, layout_y


class BoundaryGrid:
    """
    Parameterize the wind farm layout with a grid or the boundary grid method
    """

    def __init__(self, fi):
        """
        Initializes a BoundaryGrid object by assigning a
        FlorisInterface object.

        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
        """
        self.fi = fi

        self.n_boundary_turbs = 0
        self.start = 0.0
        self.nrows = 0
        self.ncols = 0
        self.farm_width = 0.0
        self.farm_height = 0.0
        self.shear = 0.0
        self.rotation = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.shrink_boundary = 0.0
        self.boundary_x = np.array([])
        self.boundary_y = np.array([])
        self.eps = 1e-3

    def reinitialize_bg(
        self,
        n_boundary_turbs=None,
        start=None,
        nrows=None,
        ncols=None,
        farm_width=None,
        farm_height=None,
        shear=None,
        rotation=None,
        center_x=None,
        center_y=None,
        shrink_boundary=None,
        boundary_x=None,
        boundary_y=None,
        eps=None,
    ):

        if n_boundary_turbs is not None:
            self.n_boundary_turbs = n_boundary_turbs
        if start is not None:
            self.start = start
        if nrows is not None:
            self.nrows = nrows
        if ncols is not None:
            self.ncols = ncols
        if farm_width is not None:
            self.farm_width = farm_width
        if farm_height is not None:
            self.farm_height = farm_height
        if shear is not None:
            self.shear = shear
        if rotation is not None:
            self.rotation = rotation
        if center_x is not None:
            self.center_x = center_x
        if center_y is not None:
            self.center_y = center_y
        if shrink_boundary is not None:
            self.shrink_boundary = shrink_boundary
        if boundary_x is not None:
            self.boundary_x = boundary_x
        if boundary_y is not None:
            self.boundary_y = boundary_y
        if eps is not None:
            self.eps = eps

    def reinitialize_xy(self):

        layout_x, layout_y = boundary_grid(
            self.n_boundary_turbs,
            self.start,
            self.nrows,
            self.ncols,
            self.farm_width,
            self.farm_height,
            self.shear,
            self.rotation,
            self.center_x,
            self.center_y,
            self.shrink_boundary,
            self.boundary_x,
            self.boundary_y,
            eps=self.eps,
        )

        self.fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))


if __name__ == "__main__":

    nrows = 10
    ncols = 10
    farm_width = 600
    farm_height = 600
    shear = np.deg2rad(10)
    rotation = np.deg2rad(30)
    center_x = 250
    center_y = 300
    boundary_mult = 0.6
    shrink_boundary = 10.0
    # boundary_x = np.array([-300.0,-100.0,100.0,100.0,0.0]) + 2200.0
    # boundary_y = np.array([-100.0,100.0,140.0,-100.0,0.0]) + 300.0
    boundary_x = np.array(
        [0.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0, 400.0, 400.0, 500.0, 500.0, 0.0]
    )
    boundary_y = np.array(
        [
            500.0,
            500.0,
            400.0,
            400.0,
            300.0,
            300.0,
            200.0,
            200.0,
            100.0,
            100.0,
            600.0,
            600.0,
        ]
    )
    x, y = discontinuous_grid(
        nrows,
        ncols,
        farm_width,
        farm_height,
        shear,
        rotation,
        center_x,
        center_y,
        shrink_boundary,
        boundary_x,
        boundary_y,
    )

    n_boundary_turbs = 25
    start = 1000.0
    layout_x, layout_y = place_boundary_turbines(
        n_boundary_turbs, start, boundary_x, boundary_y
    )

    bx = np.append(boundary_x, boundary_x[0])
    by = np.append(boundary_y, boundary_y[0])

    boundary = np.zeros((len(bx), 2))
    boundary[:, 0] = bx[:]
    boundary[:, 1] = by[:]
    poly = Polygon(boundary)

    # centroid = poly.centroid
    # new_bx = (bx[:]-centroid.x)*boundary_mult + centroid.x
    # new_by = (by[:]-centroid.y)*boundary_mult + centroid.y

    nBounds = len(bx)
    new_bx = np.zeros(len(bx))
    new_by = np.zeros(len(by))
    eps = 1e-3
    for i in range(nBounds):
        point = Point(bx[i] + eps, by[i])
        if poly.contains(point) is True or poly.touches(point) is True:
            new_bx[i] = bx[i] + shrink_boundary
        else:
            new_bx[i] = bx[i] - shrink_boundary

        point = Point(bx[i], by[i] + eps)
        if poly.contains(point) is True or poly.touches(point) is True:
            new_by[i] = by[i] + shrink_boundary
        else:
            new_by[i] = by[i] - shrink_boundary

    import matplotlib.pyplot as plt

    nx, ny = boundary_grid(
        n_boundary_turbs,
        start,
        nrows,
        ncols,
        farm_width,
        farm_height,
        shear,
        rotation,
        center_x,
        center_y,
        shrink_boundary,
        boundary_x,
        boundary_y,
    )

    plt.plot(bx, by)
    plt.plot(new_bx, new_by)
    plt.plot(x, y, "o")
    plt.plot(layout_x, layout_y, "o")
    plt.axis("equal")

    plt.figure(2)
    plt.plot(bx, by)
    plt.plot(nx, ny, "o")
    plt.axis("equal")

    plt.show()
