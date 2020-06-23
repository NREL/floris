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


import numpy as np
import matplotlib.pyplot as plt


class Layout:
    def __init__(self, fi, boundaries, wdir=None, wspd=None, wfreq=None):
        self.fi = fi
        self.boundaries = boundaries

        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])
        self.x0 = self.fi.layout_x
        self.y0 = self.fi.layout_y

        self.min_dist = 2 * self.rotor_diameter

        if wdir is not None:
            self.wdir = wdir
        else:
            self.wdir = self.fi.floris.farm.flow_field.wind_direction
        if wspd is not None:
            self.wspd = wspd
        else:
            self.wspd = self.fi.floris.farm.flow_field.wind_speed
        if wfreq is not None:
            self.wfreq = wfreq
        else:
            self.wfreq = 1.0

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
        self.fi.reinitialize_flow_field(layout_array=[self.x, self.y])

        # Compute the objective function
        funcs = {}
        funcs["obj"] = (
            -1 * self.fi.get_farm_AEP(self.wdir, self.wspd, self.wfreq) * 1e-9
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
        self.x = varDict["x"]
        self.y = varDict["y"]

    def parse_sol_vars(self, sol):
        self.x = list(sol.getDVs().values())[0]
        self.y = list(sol.getDVs().values())[1]

    def add_var_group(self, optProb):
        optProb.addVarGroup(
            "x", self.nturbs, type="c", lower=self.xmin, upper=self.xmax, value=self.x0
        )
        optProb.addVarGroup(
            "y", self.nturbs, type="c", lower=self.ymin, upper=self.ymax, value=self.y0
        )

        return optProb

    def add_con_group(self, optProb):
        optProb.addConGroup("boundary_con", self.nturbs, lower=0.0)
        optProb.addConGroup("spacing_con", self.nturbs, lower=self.min_dist)

        return optProb

    def compute_cons(self, funcs):
        funcs["boundary_con"] = self.distance_from_boundaries()
        funcs["spacing_con"] = self.space_constraint()

        return funcs

    ###########################################################################
    # User-defined methods
    ###########################################################################

    def space_constraint(self):
        dist = [
            np.min(
                [
                    np.sqrt((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2)
                    for j in range(self.nturbs)
                    if i != j
                ]
            )
            for i in range(self.nturbs)
        ]

        return dist

    def distance_from_boundaries(self):
        x = self.x
        y = self.y

        dist_out = []

        for k in range(self.nturbs):
            dist = []
            in_poly = self.point_inside_polygon(self.x[k], self.y[k], self.boundaries)

            for i in range(len(self.boundaries)):
                self.boundaries = np.array(self.boundaries)
                p1 = self.boundaries[i]
                if i == len(self.boundaries) - 1:
                    p2 = self.boundaries[0]
                else:
                    p2 = self.boundaries[i + 1]

                px = p2[0] - p1[0]
                py = p2[1] - p1[1]
                norm = px * px + py * py

                u = (
                    (self.x[k] - self.boundaries[i][0]) * px
                    + (self.y[k] - self.boundaries[i][1]) * py
                ) / float(norm)

                if u <= 0:
                    xx = p1[0]
                    yy = p1[1]
                elif u >= 1:
                    xx = p2[0]
                    yy = p2[1]
                else:
                    xx = p1[0] + u * px
                    yy = p1[1] + u * py

                dx = self.x[k] - xx
                dy = self.y[k] - yy
                dist.append(np.sqrt(dx * dx + dy * dy))

            dist = np.array(dist)
            if in_poly:
                dist_out.append(np.min(dist))
            else:
                dist_out.append(-np.min(dist))

        dist_out = np.array(dist_out)

        return dist_out

    def point_inside_polygon(self, x, y, poly):
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def plot_layout_opt_results(self, sol):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx = sol.getDVs()["x"]
        locsy = sol.getDVs()["y"]

        plt.figure(figsize=(9, 6))
        fontsize = 16
        plt.plot(self.x0, self.y0, "ob")
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
        self._nturbs = len(self.fi.floris.farm.turbines)
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fi.floris.farm.turbine_map.turbines[0].rotor_diameter
