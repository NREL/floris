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
from shapely.geometry import Point

from .layout_optimization_base import LayoutOptimization


class LayoutOptimizationPyOptSparse(LayoutOptimization):
    def __init__(
        self,
        fi,
        boundaries,
        min_dist=None,
        freq=None,
        solver=None,
        optOptions=None,
        timeLimit=None,
        storeHistory='hist.hist',
        hotStart=None
    ):
        super().__init__(fi, boundaries, min_dist=min_dist, freq=freq)
        self._reinitialize(solver=solver, optOptions=optOptions)

        self.storeHistory = storeHistory
        self.timeLimit = timeLimit
        self.hotStart = hotStart

    def _reinitialize(self, solver=None, optOptions=None):
        try:
            import pyoptsparse
        except ImportError:
            err_msg = (
                "It appears you do not have pyOptSparse installed. "
                + "Please refer to https://pyoptsparse.readthedocs.io/ for "
                + "guidance on how to properly install the module."
            )
            self.logger.error(err_msg, stack_info=True)
            raise ImportError(err_msg)

        # Insantiate ptOptSparse optimization object with name and objective function
        self.optProb = pyoptsparse.Optimization('layout', self._obj_func)

        self.optProb = self.add_var_group(self.optProb)
        self.optProb = self.add_con_group(self.optProb)
        self.optProb.addObj("obj")

        if solver is not None:
            self.solver = solver
            print("Setting up optimization with user's choice of solver: ", self.solver)
        else:
            self.solver = "SLSQP"
            print("Setting up optimization with default solver: SLSQP.")
        if optOptions is not None:
            self.optOptions = optOptions
        else:
            if self.solver == "SNOPT":
                self.optOptions = {"Major optimality tolerance": 1e-7}
            else:
                self.optOptions = {}

        exec("self.opt = pyoptsparse." + self.solver + "(options=self.optOptions)")

    def _optimize(self):
        if hasattr(self, "_sens"):
            self.sol = self.opt(self.optProb, sens=self._sens)
        else:
            if self.timeLimit is not None:
                self.sol = self.opt(
                    self.optProb,
                    sens="CDR",
                    storeHistory=self.storeHistory,
                    timeLimit=self.timeLimit,
                    hotStart=self.hotStart
                )
            else:
                self.sol = self.opt(
                    self.optProb,
                    sens="CDR",
                    storeHistory=self.storeHistory,
                    hotStart=self.hotStart
                )
        return self.sol

    def _obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Update turbine map with turbince locations
        # self.fi.reinitialize(layout=[self.x, self.y])
        # self.fi.calculate_wake()

        # Compute the objective function
        funcs = {}
        funcs["obj"] = (
            -1 * self.mean_distance(self.x, self.y)
           #  -1 * np.sum(self.fi.get_farm_power() * self.freq * 8760) / self.initial_AEP
        )

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs, self.x, self.y)

        fail = False
        return funcs, fail

    # Optionally, the user can supply the optimization with gradients
    # def _sens(self, varDict, funcs):
    #     funcsSens = {}
    #     fail = False
    #     return funcsSens, fail

    def parse_opt_vars(self, varDict):
        self.x = self._unnorm(varDict["x"], self.xmin, self.xmax)
        self.y = self._unnorm(varDict["y"], self.ymin, self.ymax)

    def parse_sol_vars(self, sol):
        self.x = list(self._unnorm(sol.getDVs()["x"], self.xmin, self.xmax))[0]
        self.y = list(self._unnorm(sol.getDVs()["y"], self.ymin, self.ymax))[1]

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

    def compute_cons(self, funcs, x, y):
        funcs["boundary_con"] = self.distance_from_boundaries(x, y)
        funcs["spacing_con"] = self.space_constraint(x, y)

        return funcs

    def mean_distance(self, x, y):

        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        return np.mean(distances)


    def space_constraint(self, x, y, rho=500):
        # Calculate distances between turbines
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

    def distance_from_boundaries(self, x, y):
        boundary_con = np.zeros(self.nturbs)
        for i in range(self.nturbs):
            loc = Point(x[i], y[i])
            boundary_con[i] = loc.distance(self.boundary_line)
            if self.boundary_polygon.contains(loc) is True:
                boundary_con[i] *= -1.0

        return boundary_con

    def plot_layout_opt_results(self):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx = self._unnorm(self.sol.getDVs()["x"], self.xmin, self.xmax)
        locsy = self._unnorm(self.sol.getDVs()["y"], self.ymin, self.ymax)
        x0 = self._unnorm(self.x0, self.xmin, self.xmax)
        y0 = self._unnorm(self.y0, self.ymin, self.ymax)

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
