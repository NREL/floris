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
from scipy.optimize import minimize

from .optimization import Optimization


class LayoutOptimization(Optimization):
    """
    Layout is a subclass of the
    :py:class:`~.tools.optimization.scipy.optimization.Optimization` class
    that is used to perform layout optimization.
    """

    def __init__(
        self,
        fi,
        boundaries,
        wd,
        ws,
        freq,
        AEP_initial,
        x0=None,
        bnds=None,
        min_dist=None,
        opt_method="SLSQP",
        opt_options=None,
    ):
        """
        Instantiate LayoutOptimization object with a FlorisInterface object and
        assign parameter values.

        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            wd (np.array): An array of wind directions (deg).
            ws (np.array): An array of wind speeds (m/s).
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values.
            AEP_initial (float): The initial Annual Energy
                Production used for normalization in the optimization (Wh)
                (TODO: Is Watt-hours the correct unit?).
            x0 (iterable, optional): The initial turbine locations,
                ordered by x-coordinate and then y-coordiante
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]) (m). If none are
                provided, x0 initializes to the current turbine locations.
                Defaults to None.
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to the min. and max. of the
                boundaries iterable. Defaults to None.
            min_dist (float, optional): The minimum distance to be
                maintained between turbines during the optimization (m). If not
                specified, initializes to 2 rotor diameters. Defaults to None.
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to 'SLSQP'.
            opt_options (dict, optional): Optimization options used by
                scipy.optimize.minize. If none are specified, they are set to
                {'maxiter': 100, 'disp': True, 'iprint': 2, 'ftol': 1e-9}.
                Defaults to None.
        """
        super().__init__(fi)
        self.epsilon = np.finfo(float).eps

        if opt_options is None:
            self.opt_options = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-9}

        self.reinitialize_opt(
            boundaries=boundaries,
            wd=wd,
            ws=ws,
            freq=freq,
            AEP_initial=AEP_initial,
            x0=x0,
            bnds=bnds,
            min_dist=min_dist,
            opt_method=opt_method,
            opt_options=opt_options,
        )

    # Private methods

    def _AEP_layout_opt(self, locs):
        locs_unnorm = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in locs[0 : self.nturbs]
        ] + [
            self._unnorm(valy, self.bndy_min, self.bndy_max)
            for valy in locs[self.nturbs : 2 * self.nturbs]
        ]
        self._change_coordinates(locs_unnorm)
        AEP_sum = self._AEP_loop_wd()
        return -1 * AEP_sum / self.AEP_initial

    def _AEP_single_wd(self, wd, ws, freq):
        self.fi.reinitialize_flow_field(wind_direction=[wd], wind_speed=[ws])
        self.fi.calculate_wake()

        turb_powers = [turbine.power for turbine in self.fi.floris.farm.turbines]
        return np.sum(turb_powers) * freq * 8760

    def _AEP_loop_wd(self):
        AEP_sum = 0

        for i in range(len(self.wd)):
            self.fi.reinitialize_flow_field(
                wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
            )
            self.fi.calculate_wake()

            AEP_sum = AEP_sum + self.fi.get_farm_power() * self.freq[i] * 8760
        return AEP_sum

    def _change_coordinates(self, locs):
        # Parse the layout coordinates
        layout_x = locs[0 : self.nturbs]
        layout_y = locs[self.nturbs : 2 * self.nturbs]
        layout_array = [layout_x, layout_y]

        # Update the turbine map in floris
        self.fi.reinitialize_flow_field(layout_array=layout_array)

    def _space_constraint(self, x_in, min_dist):
        x = np.nan_to_num(x_in[0 : self.nturbs])
        y = np.nan_to_num(x_in[self.nturbs :])

        dist = [
            np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            for i in range(self.nturbs)
            for j in range(self.nturbs)
            if i != j
        ]

        # dist = []
        # for i in range(self.nturbs):
        #     for j in range(self.nturbs):
        #         if i != j:
        #             dist.append(np.sqrt( (x[i]-x[j])**2 + (y[i]-y[j])**2))

        return np.min(dist) - self._norm(min_dist, self.bndx_min, self.bndx_max)

    def _distance_from_boundaries(self, x_in, boundaries):
        # x = self._unnorm(x_in[0:self.nturbs], self.bndx_min, self.bndx_max)
        # y = self._unnorm(x_in[self.nturbs:2*self.nturbs], \
        #                  self.bndy_min, self.bndy_max)
        x = x_in[0 : self.nturbs]
        y = x_in[self.nturbs : 2 * self.nturbs]

        dist_out = []

        for k in range(self.nturbs):
            dist = []
            in_poly = self._point_inside_polygon(x[k], y[k], boundaries)

            for i in range(len(boundaries)):
                boundaries = np.array(boundaries)
                p1 = boundaries[i]
                if i == len(boundaries) - 1:
                    p2 = boundaries[0]
                else:
                    p2 = boundaries[i + 1]

                px = p2[0] - p1[0]
                py = p2[1] - p1[1]
                norm = px * px + py * py

                u = (
                    (x[k] - boundaries[i][0]) * px + (y[k] - boundaries[i][1]) * py
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

                dx = x[k] - xx
                dy = y[k] - yy
                dist.append(np.sqrt(dx * dx + dy * dy))

            dist = np.array(dist)
            if in_poly:
                dist_out.append(np.min(dist))
            else:
                dist_out.append(-np.min(dist))

        dist_out = np.array(dist_out)

        return np.min(dist_out)

    def _point_inside_polygon(self, x, y, poly):
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

    def _generate_constraints(self):
        # grad_constraint1 = grad(self._space_constraint)
        # grad_constraint2 = grad(self._distance_from_boundaries)

        tmp1 = {
            "type": "ineq",
            "fun": lambda x, *args: self._space_constraint(x, self.min_dist),
            "args": (self.min_dist,),
        }
        tmp2 = {
            "type": "ineq",
            "fun": lambda x, *args: self._distance_from_boundaries(
                x, self.boundaries_norm
            ),
            "args": (self.boundaries_norm,),
        }

        self.cons = [tmp1, tmp2]

    def _optimize(self):
        self.residual_plant = minimize(
            self._AEP_layout_opt,
            self.x0,
            method=self.opt_method,
            bounds=self.bnds,
            constraints=self.cons,
            options=self.opt_options,
        )

        opt_results = self.residual_plant.x

        return opt_results

    def _set_opt_bounds(self):
        self.bnds = [(0.0, 1.0) for _ in range(2 * self.nturbs)]

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

        print("Optimization complete!")

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
        wd=None,
        ws=None,
        freq=None,
        AEP_initial=None,
        x0=None,
        bnds=None,
        min_dist=None,
        opt_method=None,
        opt_options=None,
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
        if wd is not None:
            self.wd = wd
        if ws is not None:
            self.ws = ws
        if freq is not None:
            self.freq = freq
        if AEP_initial is not None:
            self.AEP_initial = AEP_initial
        else:
            self.AEP_initial = self.fi.get_farm_AEP(self.wd, self.ws, self.freq)
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [
                self._norm(coord.x1, self.bndx_min, self.bndx_max)
                for coord in self.fi.floris.farm.turbine_map.coords
            ] + [
                self._norm(coord.x2, self.bndy_min, self.bndy_max)
                for coord in self.fi.floris.farm.turbine_map.coords
            ]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 2 * self.fi.floris.farm.turbines[0].rotor_diameter
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options

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
