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


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)

from .layout_optimization_base import LayoutOptimization


class LayoutOptimizationBoundaryGrid(LayoutOptimization):
    def __init__(
        self,
        fi,
        boundaries,
        start,
        x_spacing,
        y_spacing,
        shear,
        rotation,
        center_x,
        center_y,
        boundary_setback,
        n_boundary_turbines=None,
        boundary_spacing=None,
    ):
        self.fi = fi

        self.boundary_x = np.array([val[0] for val in boundaries])
        self.boundary_y = np.array([val[1] for val in boundaries])
        boundary = np.zeros((len(self.boundary_x), 2))
        boundary[:, 0] = self.boundary_x[:]
        boundary[:, 1] = self.boundary_y[:]
        self._boundary_polygon = Polygon(boundary)

        self.start = start
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.shear = shear
        self.rotation = rotation
        self.center_x = center_x
        self.center_y = center_y
        self.boundary_setback = boundary_setback
        self.n_boundary_turbines = n_boundary_turbines
        self.boundary_spacing = boundary_spacing

    def _discontinuous_grid(
        self,
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

    def _discrete_grid(
        self,
        x_spacing,
        y_spacing,
        shear,
        rotation,
        center_x,
        center_y,
        boundary_setback,
        boundary_poly
    ):
        """
        returns grid turbine layout. Assumes the turbines fill the entire plant area

        Args:
        x_spacing (Float): grid spacing in the unrotated x direction (m)
        y_spacing (Float): grid spacing in the unrotated y direction (m)
        shear (Float): grid shear (rad)
        rotation (Float): grid rotation (rad)
        center_x (Float): the x coordinate of the grid center (m)
        center_y (Float): the y coordinate of the grid center (m)
        boundary_poly (Polygon): a shapely Polygon of the wind plant boundary

        Returns
        return_x (Array(Float)): turbine x locations
        return_y (Array(Float)): turbine y locations
        """

        shrunk_poly = boundary_poly.buffer(-boundary_setback)
        if shrunk_poly.area <= 0:
            return np.array([]), np.array([])
        # create grid
        minx, miny, maxx, maxy = shrunk_poly.bounds
        width = maxx-minx
        height = maxy-miny

        center_point = Point((center_x,center_y))
        poly_to_center = center_point.distance(shrunk_poly.centroid)

        width = np.max([width,poly_to_center])
        height = np.max([height,poly_to_center])
        nrows = int(np.max([width,height])/np.min([x_spacing,y_spacing]))*2 + 1
        ncols = nrows

        xlocs = np.arange(0,ncols)*x_spacing
        ylocs = np.arange(0,nrows)*y_spacing
        row_number = np.arange(0,nrows)

        d = np.array([i for x in xlocs for i in row_number])
        layout_x = np.array([x for x in xlocs for y in ylocs]) + d*y_spacing*np.tan(shear)
        layout_y = np.array([y for x in xlocs for y in ylocs])

        # rotate
        rotate_x = np.cos(rotation)*layout_x - np.sin(rotation)*layout_y
        rotate_y = np.sin(rotation)*layout_x + np.cos(rotation)*layout_y

        # move center of grid
        rotate_x = (rotate_x - np.mean(rotate_x)) + center_x
        rotate_y = (rotate_y - np.mean(rotate_y)) + center_y

        # get rid of points outside of boundary polygon
        meets_constraints = np.zeros(len(rotate_x),dtype=bool)
        for i in range(len(rotate_x)):
            pt = Point(rotate_x[i],rotate_y[i])
            if shrunk_poly.contains(pt) or shrunk_poly.touches(pt):
                meets_constraints[i] = True

        # arrange final x,y points
        return_x = rotate_x[meets_constraints]
        return_y = rotate_y[meets_constraints]

        return return_x, return_y

    def find_lengths(self, x, y, npoints):
        length = np.zeros(len(x) - 1)
        for i in range(npoints):
            length[i] = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
        return length

    # def _place_boundary_turbines(self, n_boundary_turbs, start, boundary_x, boundary_y):
    #     """
    #     Place turbines equally spaced traversing the perimiter if the wind farm along the boundary

    #     Args:
    #     n_boundary_turbs (Int): number of turbines to be placed on the boundary
    #     start (Float): where the first turbine should be placed
    #     boundary_x (Array(Float)): x boundary points
    #     boundary_y (Array(Float)): y boundary points

    #     Returns
    #     layout_x (Array(Float)): turbine x locations
    #     layout_y (Array(Float)): turbine y locations
    #     """

    #     # check if the boundary is closed, correct if not
    #     if boundary_x[-1] != boundary_x[0] or boundary_y[-1] != boundary_y[0]:
    #         boundary_x = np.append(boundary_x, boundary_x[0])
    #         boundary_y = np.append(boundary_y, boundary_y[0])

    #     # make the boundary
    #     boundary = np.zeros((len(boundary_x), 2))
    #     boundary[:, 0] = boundary_x[:]
    #     boundary[:, 1] = boundary_y[:]
    #     poly = Polygon(boundary)
    #     perimeter = poly.length

    #     # get the flattened turbine locations
    #     spacing = perimeter / float(n_boundary_turbs)
    #     flattened_locs = np.linspace(start, perimeter + start - spacing, n_boundary_turbs)

    #     # set all of the flattened values between 0 and the perimeter
    #     for i in range(n_boundary_turbs):
    #         while flattened_locs[i] < 0.0:
    #             flattened_locs[i] += perimeter
    #         if flattened_locs[i] > perimeter:
    #             flattened_locs[i] = flattened_locs[i] % perimeter

    #     # place the turbines around the perimeter
    #     nBounds = len(boundary_x)
    #     layout_x = np.zeros(n_boundary_turbs)
    #     layout_y = np.zeros(n_boundary_turbs)

    #     lenBound = np.zeros(nBounds - 1)
    #     for i in range(nBounds - 1):
    #         lenBound[i] = Point(boundary[i]).distance(Point(boundary[i + 1]))
    #     for i in range(n_boundary_turbs):
    #         for j in range(nBounds - 1):
    #             if flattened_locs[i] < sum(lenBound[0 : j + 1]):
    #                 layout_x[i] = (
    #                     boundary_x[j]
    #                     + (boundary_x[j + 1] - boundary_x[j])
    #                     * (flattened_locs[i] - sum(lenBound[0:j]))
    #                     / lenBound[j]
    #                 )
    #                 layout_y[i] = (
    #                     boundary_y[j]
    #                     + (boundary_y[j + 1] - boundary_y[j])
    #                     * (flattened_locs[i] - sum(lenBound[0:j]))
    #                     / lenBound[j]
    #                 )
    #                 break

    #     return layout_x, layout_y

    def _place_boundary_turbines(self, start, boundary_poly, nturbs=None, spacing=None):
        xBounds, yBounds = boundary_poly.boundary.coords.xy

        if xBounds[-1] != xBounds[0]:
            xBounds = np.append(xBounds, xBounds[0])
            yBounds = np.append(yBounds, yBounds[0])

        nBounds = len(xBounds)
        lenBound = self.find_lengths(xBounds, yBounds, len(xBounds) - 1)
        circumference = sum(lenBound)

        if nturbs is not None and spacing is None:
            # When the number of boundary turbines is specified
            nturbs = int(nturbs)
            bound_loc = np.linspace(
                start, start + circumference - circumference / float(nturbs), nturbs
            )
        elif spacing is not None and nturbs is None:
            # When the spacing of boundary turbines is specified
            nturbs = int(np.floor(circumference / spacing))
            bound_loc = np.linspace(
                start, start + circumference - circumference / float(nturbs), nturbs
            )
        else:
            raise ValueError("Please specify either nturbs or spacing.")

        x = np.zeros(nturbs)
        y = np.zeros(nturbs)

        if spacing is None:
            # When the number of boundary turbines is specified
            for i in range(nturbs):
                if bound_loc[i] > circumference:
                    bound_loc[i] = bound_loc[i] % circumference
                while bound_loc[i] < 0.0:
                    bound_loc[i] += circumference
            for i in range(nturbs):
                done = False
                for j in range(nBounds):
                    if done is False:
                        if bound_loc[i] < sum(lenBound[0:j+1]):
                            point_x = (
                                xBounds[j]
                                + (xBounds[j+1] - xBounds[j])
                                * (bound_loc[i] - sum(lenBound[0:j]))
                                / lenBound[j]
                            )
                            point_y = (
                                yBounds[j]
                                + (yBounds[j+1] - yBounds[j])
                                * (bound_loc[i] - sum(lenBound[0:j]))
                                / lenBound[j]
                            )
                            done = True
                            x[i] = point_x
                            y[i] = point_y
        else:
            # When the spacing of boundary turbines is specified
            additional_space = 0.0
            end_loop = False
            for i in range(nturbs):
                done = False
                for j in range(nBounds):
                    while done is False:
                        dist = start + i*spacing + additional_space
                        if dist < sum(lenBound[0:j+1]):
                            point_x = (
                                xBounds[j]
                                + (xBounds[j+1]-xBounds[j])
                                * (dist -sum(lenBound[0:j]))
                                / lenBound[j]
                            )
                            point_y = (
                                yBounds[j]
                                + (yBounds[j+1]-yBounds[j])
                                * (dist -sum(lenBound[0:j]))
                                / lenBound[j]
                            )

                            # Check if turbine is too close to previous turbine
                            if i > 0:
                                # Check if turbine just placed is to close to first turbine
                                min_dist = cdist([(point_x, point_y)], [(x[0], y[0])])
                                if min_dist < spacing:
                                    # TODO: make this more robust;
                                    # pass is needed if 2nd turbine is too close to the first
                                    if i == 1:
                                        pass
                                    else:
                                        end_loop = True
                                        ii = i
                                        break

                                min_dist = cdist([(point_x, point_y)], [(x[i-1], y[i-1])])
                                if min_dist < spacing:
                                    additional_space += 1.0
                                else:
                                    done = True
                                    x[i] = point_x
                                    y[i] = point_y
                            elif i == 0:
                                # If first turbine, just add initial turbine point
                                done = True
                                x[i] = point_x
                                y[i] = point_y
                            else:
                                pass
                        else:
                            break
                    if end_loop is True:
                        break
                if end_loop is True:
                    x = x[:ii]
                    y = y[:ii]
                    break
        return x, y

    def _place_boundary_turbines_with_specified_spacing(
        self,
        spacing,
        start,
        boundary_x,
        boundary_y
    ):
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
        n_boundary_turbs = int(perimeter / float(spacing))
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
        self,
        start,
        x_spacing,
        y_spacing,
        shear,
        rotation,
        center_x,
        center_y,
        boundary_setback,
        n_boundary_turbines=None,
        boundary_spacing=None,
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

        boundary_turbines_x, boundary_turbines_y = self._place_boundary_turbines(
            start, self._boundary_polygon, nturbs=n_boundary_turbines, spacing=boundary_spacing
        )
        # ( boundary_turbines_x,
        #  boundary_turbines_y ) = self._place_boundary_turbines_with_specified_spacing(
        #     spacing, start, boundary_x, boundary_y
        # )

        # grid_turbines_x, grid_turbines_y = self._discontinuous_grid(
        #     nrows,
        #     ncols,
        #     farm_width,
        #     farm_height,
        #     shear,
        #     rotation,
        #     center_x,
        #     center_y,
        #     shrink_boundary,
        #     boundary_x,
        #     boundary_y,
        #     eps=eps,
        # )

        grid_turbines_x, grid_turbines_y = self._discrete_grid(
            x_spacing,
            y_spacing,
            shear,
            rotation,
            center_x,
            center_y,
            boundary_setback,
            self._boundary_polygon,
        )

        layout_x = np.append(boundary_turbines_x, grid_turbines_x)
        layout_y = np.append(boundary_turbines_y, grid_turbines_y)

        return layout_x, layout_y

    def reinitialize_bg(
        self,
        n_boundary_turbines=None,
        start=None,
        x_spacing=None,
        y_spacing=None,
        shear=None,
        rotation=None,
        center_x=None,
        center_y=None,
        boundary_setback=None,
        boundary_x=None,
        boundary_y=None,
        boundary_spacing=None,
    ):

        if n_boundary_turbines is not None:
            self.n_boundary_turbines = n_boundary_turbines
        if start is not None:
            self.start = start
        if x_spacing is not None:
            self.x_spacing = x_spacing
        if y_spacing is not None:
            self.y_spacing = y_spacing
        if shear is not None:
            self.shear = shear
        if rotation is not None:
            self.rotation = rotation
        if center_x is not None:
            self.center_x = center_x
        if center_y is not None:
            self.center_y = center_y
        if boundary_setback is not None:
            self.boundary_setback = boundary_setback
        if boundary_x is not None:
            self.boundary_x = boundary_x
        if boundary_y is not None:
            self.boundary_y = boundary_y
        if boundary_spacing is not None:
            self.boundary_spacing = boundary_spacing

    def reinitialize_xy(self):
        layout_x, layout_y = self.boundary_grid(
            self.start,
            self.x_spacing,
            self.y_spacing,
            self.shear,
            self.rotation,
            self.center_x,
            self.center_y,
            self.boundary_setback,
            self.n_boundary_turbines,
            self.boundary_spacing,
        )

        self.fi.reinitialize(layout=(layout_x, layout_y))

    def plot_layout(self):
        plt.figure(figsize=(9, 6))
        fontsize = 16

        plt.plot(self.fi.layout_x, self.fi.layout_y, "ob")
        # plt.plot(locsx, locsy, "or")

        plt.xlabel("x (m)", fontsize=fontsize)
        plt.ylabel("y (m)", fontsize=fontsize)
        plt.axis("equal")
        plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)

        plt.show()

    def space_constraint(self, x, y, min_dist, rho=500):
        # Calculate distances between turbines
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)

        g = 1 - np.array(dist) / min_dist

        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)

        return KS_constraint[0][0], dist
