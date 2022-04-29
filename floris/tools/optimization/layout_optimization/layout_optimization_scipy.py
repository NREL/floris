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
from scipy.optimize import minimize
from shapely.geometry import Point
from scipy.spatial.distance import cdist

from .layout_optimization_base import LayoutOptimization

class LayoutOptimizationScipy(LayoutOptimization):
    def __init__(
        self,
        fi,
        boundaries,
        freq=None,
        x0=None,
        bnds=None,
        min_dist=None,
        solver='SLSQP',
        optOptions=None,
    ):
        super().__init__(fi, boundaries, min_dist=min_dist, freq=freq)

        if optOptions is None:
            self.optOptions = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-9, "eps":0.01}

        self.reinitialize_opt(
            boundaries=boundaries,
            freq=freq,
            x0=x0,
            bnds=bnds,
            solver=solver,
            optOptions=self.optOptions,
        )


    # Private methods

    def _optimize(self):
        self.residual_plant = minimize(
            self._obj_func,
            self.x0,
            method=self.solver,
            bounds=self.bnds,
            constraints=self.cons,
            options=self.optOptions,
        )

        return self.residual_plant.x

    def _obj_func(self, locs):
        locs_unnorm = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in locs[0 : self.nturbs]
        ] + [
            self._unnorm(valy, self.bndy_min, self.bndy_max)
            for valy in locs[self.nturbs : 2 * self.nturbs]
        ]
        self._change_coordinates(locs_unnorm)
        self.fi.calculate_wake()
        AEP_sum = np.sum(self.fi.get_farm_power() * self.freq * 8760)
        return -1 * AEP_sum / self.initial_AEP

    def _change_coordinates(self, locs):
        # Parse the layout coordinates
        layout_x = locs[0 : self.nturbs]
        layout_y = locs[self.nturbs : 2 * self.nturbs]
        layout_array = (layout_x, layout_y)

        # Update the turbine map in floris
        self.fi.reinitialize(layout=layout_array)

    def _generate_constraints(self):
        # grad_constraint1 = grad(self._space_constraint)
        # grad_constraint2 = grad(self._distance_from_boundaries)

        tmp1 = {
            "type": "ineq",
            "fun": lambda x, *args: self._space_constraint(x),
        }
        tmp2 = {
            "type": "ineq",
            "fun": lambda x: self._distance_from_boundaries(x),
        }

        self.cons = [tmp1, tmp2]

    def _set_opt_bounds(self):
        self.bnds = [(0.0, 1.0) for _ in range(2 * self.nturbs)]

    def _space_constraint(self, x_in):
        rho=500
        x = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in x_in[0 : self.nturbs]
        ] 
        y =  [
            self._unnorm(valy, self.bndy_min, self.bndy_max)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]
        # x = np.array(x_in[0 : self.nturbs])
        # y = np.array(x_in[self.nturbs :])
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

        return -1*KS_constraint[0][0]

    def _distance_from_boundaries(self, x_in):
        # x = np.array(x_in[0 : self.nturbs])
        # y = np.array(x_in[self.nturbs :])
        x = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in x_in[0 : self.nturbs]
        ] 
        y =  [
            self._unnorm(valy, self.bndy_min, self.bndy_max)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]
        boundary_con = np.zeros(self.nturbs)
        for i in range(self.nturbs):
            loc = Point(x[i], y[i])
            boundary_con[i] = loc.distance(self.boundary_line)
            if self.boundary_polygon.contains(loc)==True:
                boundary_con[i] *= 1.0

        return boundary_con

    # Public methods

    def optimize(self):
        """
        This method finds the optimized layout of wind turbines for power
        production given the provided frequencies of occurance of wind
        conditions (wind speed, direction).

        Returns:
            opt_locs (iterable): A list of the optimized locations of each
            turbine (m).
        """
        print("=====================================================")
        print("Optimizing turbine layout...")
        print("Number of parameters to optimize = ", len(self.x0))
        print("=====================================================")

        opt_locs_norm = self._optimize()

        print("Optimization complete.")

        opt_locs = [
            [
                self._unnorm(valx, self.bndx_min, self.bndx_max)
                for valx in opt_locs_norm[0 : self.nturbs]
            ],
            [
                self._unnorm(valy, self.bndy_min, self.bndy_max)
                for valy in opt_locs_norm[self.nturbs : 2 * self.nturbs]
            ],
        ]

        return opt_locs

    def reinitialize_opt(
        self,
        boundaries=None,
        freq=None,
        x0=None,
        bnds=None,
        solver=None,
        optOptions=None,
    ):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.

        Args:
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            wd (np.array): An array of wind directions (deg). Defaults to None.
            ws (np.array): An array of wind speeds (m/s). Defaults to None.
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values. Defaults to None.
            AEP_initial (float): The initial Annual Energy
                Production used for normalization in the optimization (Wh). If
                not specified, initializes to the AEP of the current Floris
                object. Defaults to None.
            x0 (iterable, optional): The initial turbine locations,
                ordered by x-coordinate and then y-coordiante (ie. [x1, x2, ...
                , xn, y1, y2, ..., yn] (m)). If none are provided, x0
                initializes to the current turbine locations. Defaults to None.
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to the min. and max. of the
                boundaries iterable. Defaults to None.
            min_dist (float, optional): The minimum distance to be maintained
                between turbines during the optimization (m). If not specified,
                initializes to 2 rotor diameters. Defaults to None.
            opt_method (str, optional): The optimization method for
                scipy.optimize.minize to use. If none is specified, initializes
                to 'SLSQP'. Defaults to None.
            opt_options (dict, optional): Dicitonary for setting the
                optimization options. Defaults to None.
        """
        if boundaries is not None:
            self.boundaries = boundaries
            self.bndx_min = np.min([val[0] for val in boundaries])
            self.bndy_min = np.min([val[1] for val in boundaries])
            self.bndx_max = np.max([val[0] for val in boundaries])
            self.bndy_max = np.max([val[1] for val in boundaries])
            self.boundaries_norm = [
                [
                    self._norm(val[0], self.bndx_min, self.bndx_max),
                    self._norm(val[1], self.bndy_min, self.bndy_max),
                ]
                for val in self.boundaries
            ]
        if freq is not None:
            self.freq = freq
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [
                self._norm(x, self.bndx_min, self.bndx_max)
                for x in self.fi.layout_x
            ] + [
                self._norm(y, self.bndy_min, self.bndy_max)
                for y in self.fi.layout_y
            ]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if solver is not None:
            self.solver = solver
        if optOptions is not None:
            self.optOptions = optOptions

        self._generate_constraints()

    def plot_layout_opt_results(self):
        """
        This method plots the original and new locations of the turbines in a
        wind farm after layout optimization.
        """
        locsx_old = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in self.x0[0 : self.nturbs]
        ]
        locsy_old = [
            self._unnorm(valy, self.bndy_min, self.bndy_max)
            for valy in self.x0[self.nturbs : 2 * self.nturbs]
        ]
        locsx = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in self.residual_plant.x[0 : self.nturbs]
        ]
        locsy = [
            self._unnorm(valy, self.bndy_min, self.bndy_max)
            for valy in self.residual_plant.x[self.nturbs : 2 * self.nturbs]
        ]

        plt.figure(figsize=(9, 6))
        fontsize = 16
        plt.plot(locsx_old, locsy_old, "ob")
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
