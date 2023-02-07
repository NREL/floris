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


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)


def _norm(val, x1, x2):
        return (val - x1) / (x2 - x1)

def _unnorm(val, x1, x2):
    return np.array(val) * (x2 - x1) + x1

class Layout:
    def __init__(self, fi, boundaries, freq):
        self.fi = fi
        self.boundaries = boundaries
        self.freq = freq

        self.boundary_polygon = Polygon(self.boundaries)
        self.boundary_line = LineString(self.boundaries)

        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])
        self.x0 = _norm(self.fi.layout_x, self.xmin, self.xmax)
        self.y0 = _norm(self.fi.layout_y, self.ymin, self.ymax)

        self.min_dist = 2 * self.rotor_diameter

        self.wdir = self.fi.floris.flow_field.wind_directions
        self.wspd = self.fi.floris.flow_field.wind_speeds
        self.initial_AEP = np.sum(self.fi.get_farm_power() * self.freq)

    def __str__(self):
        return "layout"

    ###########################################################################
    # Required private optimization methods
    ###########################################################################

    def reinitialize(self):
        pass

    def obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Update turbine map with turbince locations
        self.fi.reinitialize(layout_x=self.x, layout_y=self.y)
        self.fi.calculate_wake()

        # Compute the objective function
        funcs = {}
        funcs["obj"] = (
            -1 * np.sum(self.fi.get_farm_power() * self.freq) / self.initial_AEP
        )

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs)

        fail = False
        return funcs, fail

    # Optionally, the user can supply the optimization with gradients
    # def _sens(self, varDict, funcs):
    #     funcsSens = {}
    #     fail = False
    #     return funcsSens, fail

    def parse_opt_vars(self, varDict):
        self.x = _unnorm(varDict["x"], self.xmin, self.xmax)
        self.y = _unnorm(varDict["y"], self.ymin, self.ymax)

    def parse_sol_vars(self, sol):
        self.x = list(_unnorm(sol.getDVs()["x"], self.xmin, self.xmax))[0]
        self.y = list(_unnorm(sol.getDVs()["y"], self.ymin, self.ymax))[1]

    def add_var_group(self, optProb):
        optProb.addVarGroup(
            "x", self.nturbs, type="c", lower=0.0, upper=1.0, value=self.x0
        )
        optProb.addVarGroup(
            "y", self.nturbs, type="c", lower=0.0, upper=1.0, value=self.y0
        )

        return optProb

    def add_con_group(self, optProb):
        optProb.addConGroup("boundary_con", self.nturbs, upper=0.0)
        optProb.addConGroup("spacing_con", 1, upper=0.0)

        return optProb

    def compute_cons(self, funcs):
        funcs["boundary_con"] = self.distance_from_boundaries()
        funcs["spacing_con"] = self.space_constraint()

        return funcs

    ###########################################################################
    # User-defined methods
    ###########################################################################

    def space_constraint(self, rho=500):
        x = self.x
        y = self.y

        # Sped up distance calc here using vectorization
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)

        g = 1 - np.array(dist) / self.min_dist

        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)

        return KS_constraint[0][0]

    def distance_from_boundaries(self):
        boundary_con = np.zeros(self.nturbs)
        for i in range(self.nturbs):
            loc = Point(self.x[i], self.y[i])
            boundary_con[i] = loc.distance(self.boundary_line)
            if self.boundary_polygon.contains(loc) is True:
                boundary_con[i] *= -1.0

        return boundary_con

    def plot_layout_opt_results(self, sol):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx = _unnorm(sol.getDVs()["x"], self.xmin, self.xmax)
        locsy = _unnorm(sol.getDVs()["y"], self.ymin, self.ymax)
        x0 = _unnorm(self.x0, self.xmin, self.xmax)
        y0 = _unnorm(self.y0, self.ymin, self.ymax)

        plt.figure(figsize=(9, 6))
        fontsize = 16
        plt.plot(x0, y0, "ob")
        plt.plot(locsx, locsy, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel("x (m)", fontsize=fontsize)
        plt.ylabel("y (m)", fontsize=fontsize)
        plt.axis("equal")
        plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)
        plt.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts) - 1:
                plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
            else:
                plt.plot(
                    [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
                )

        plt.show()

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
