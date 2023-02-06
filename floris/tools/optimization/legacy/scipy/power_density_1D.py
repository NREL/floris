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
from scipy.optimize import minimize

from .optimization import Optimization


class PowerDensityOptimization1D(Optimization):
    """
    PowerDensityOptimization1D is a subclass of the
    :py:class:`~.tools.optimization.scipy.optimization.Optimization` class
    that performs layout optimization in 1 dimension. TODO: What is this single
    dimension?
    """

    def __init__(
        self,
        fi,
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
        Instantiate PowerDensityOptimization1D object with a FlorisInterface
        object and assigns parameter values.

        Args:
            fi (:py:class:`floris.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
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
                none are specified, they are set to some example values (TODO:
                what is the significance of these example values?). Defaults to
                None.
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
        self.counter = 0

        if opt_options is None:
            self.opt_options = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-9}

        self.reinitialize_opt(
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

    def _PowDens_opt(self, optVars):
        locs = optVars[0 : self.nturbs]
        locs_unnorm = [
            self._unnorm(valx, self.bndx_min, self.bndx_max) for valx in locs
        ]
        turb_controls = [
            optVars[self.nturbs + i * self.nturbs : 2 * self.nturbs + i * self.nturbs]
            for i in range(len(self.wd))
        ]
        turb_controls_unnorm = [
            self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in turb_controls
        ]

        self._change_coordinates(locs_unnorm)

        for i, turbine in enumerate(self.fi.floris.farm.turbine_map.turbines):
            turbine.yaw_angle = turb_controls_unnorm[0][i]

        layout_dist = self._avg_dist(locs)
        # AEP_sum = self._AEP_single_wd(self.wd[0], self.ws[0])
        # print('AEP ratio: ', AEP_sum/self.AEP_initial)

        return layout_dist / self.layout_dist_initial

    def _avg_dist(self, locs):
        dist = []
        for i in range(len(locs) - 1):
            dist.append(locs[i + 1] - locs[i])

        return np.mean(dist)

    def _change_coordinates(self, locs):
        # Parse the layout coordinates
        layout_x = locs
        layout_y = [coord.x2 for coord in self.fi.floris.farm.turbine_map.coords]
        layout_array = [layout_x, layout_y]

        # Update the turbine map in floris
        self.fi.reinitialize_flow_field(layout_array=layout_array)

    def _set_opt_bounds(self):
        # self.bnds = [(0.0, 1.0) for _ in range(2*self.nturbs)]
        self.bnds = [
            (0.0, 0.0),
            (0.083333, 0.25),
            (0.166667, 0.5),
            (0.25, 0.75),
            (0.33333, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ]

    def _AEP_single_wd(self, wd, ws):
        self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
        self.fi.calculate_wake()

        turb_powers = [turbine.power for turbine in self.fi.floris.farm.turbines]
        return np.sum(turb_powers) * self.freq[0] * 8760

    def _AEP_constraint(self, optVars):
        locs = optVars[0 : self.nturbs]
        locs_unnorm = [
            self._unnorm(valx, self.bndx_min, self.bndx_max) for valx in locs
        ]
        turb_controls = [
            optVars[self.nturbs + i * self.nturbs : 2 * self.nturbs + i * self.nturbs]
            for i in range(len(self.wd))
        ]
        turb_controls_unnorm = [
            self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in turb_controls
        ]

        for i, turbine in enumerate(self.fi.floris.farm.turbine_map.turbines):
            turbine.yaw_angle = turb_controls_unnorm[0][i]

        self._change_coordinates(locs_unnorm)

        return (
            self._AEP_single_wd(self.wd[0], self.ws[0]) / self.AEP_initial - 1
        ) * 1000000.0

    def _space_constraint(self, x_in, min_dist):
        x = np.nan_to_num(x_in[0 : self.nturbs])
        y = np.nan_to_num(x_in[self.nturbs :])

        dist = [
            np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            for i in range(self.nturbs)
            for j in range(self.nturbs)
            if i != j
        ]

        return np.min(dist) - self._norm(min_dist, self.bndx_min, self.bndx_max)

    def _generate_constraints(self):
        tmp1 = {
            "type": "ineq",
            "fun": lambda x, *args: self._space_constraint(x, self.min_dist),
            "args": (self.min_dist,),
        }
        tmp2 = {"type": "ineq", "fun": lambda x, *args: self._AEP_constraint(x)}

        self.cons = [tmp1, tmp2]

    def _optimize(self):
        self.residual_plant = minimize(
            self._PowDens_opt,
            self.x0,
            method=self.opt_method,
            bounds=self.bnds,
            constraints=self.cons,
            options=self.opt_options,
        )

        opt_results = self.residual_plant.x

        return opt_results

    def optimize(self):
        """
        This method finds the optimized layout of wind turbines for power
        production given the provided frequencies of occurance of wind
        conditions (wind speed, direction).

        Returns:
            opt_locs (iterable): A list of the optimized x, y locations of each
            turbine (m).
        """
        print("=====================================================")
        print("Optimizing turbine layout...")
        print("Number of parameters to optimize = ", len(self.x0))
        print("=====================================================")

        opt_vars_norm = self._optimize()

        print("Optimization complete.")

        opt_locs = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in opt_vars_norm[0 : self.nturbs]
        ]

        opt_yaw = [
            self._unnorm(yaw, self.yaw_min, self.yaw_max)
            for yaw in opt_vars_norm[self.nturbs :]
        ]

        return [opt_locs, opt_yaw]

    def reinitialize_opt(
        self,
        wd=None,
        ws=None,
        freq=None,
        AEP_initial=None,
        x0=None,
        bnds=None,
        min_dist=None,
        yaw_lims=None,
        opt_method=None,
        opt_options=None,
    ):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.

        Args:
            wd (np.array): An array of wind directions (deg). Defaults to None.
            ws (np.array): An array of wind speeds (m/s). Defaults to None.
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values. Defaults to None.
            AEP_initial (float): The initial Annual Energy
                Production used for normalization in the optimization (Wh)
                (TODO: Is Watt-hours the correct unit?). Defaults to None.
            x0 (iterable, optional): The initial turbine locations,
                ordered by x-coordinate and then y-coordiante
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]) (m). If none are
                provided, x0 initializes to the current turbine locations.
                Defaults to None.
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to some example values (TODO:
                what is the significance of these example values?). Defaults to
                None.
            min_dist (float, optional): The minimum distance to be
                maintained between turbines during the optimization (m). If not
                specified, initializes to 2 rotor diameters. Defaults to None.
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to None.
            opt_options (dict, optional): Optimization options used by
                scipy.optimize.minize. Defaults to None.
        """
        # if boundaries is not None:
        #     self.boundaries = boundaries
        #     self.bndx_min = np.min([val[0] for val in boundaries])
        #     self.bndy_min = np.min([val[1] for val in boundaries])
        #     self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
        #                           self.bndx_max)] for val in self.boundaries]
        self.bndx_min = np.min(
            [coord.x1 for coord in self.fi.floris.farm.turbine_map.coords]
        )
        self.bndx_max = np.max(
            [coord.x1 for coord in self.fi.floris.farm.turbine_map.coords]
        )
        if yaw_lims is not None:
            self.yaw_min = yaw_lims[0]
            self.yaw_max = yaw_lims[1]
        else:
            self.yaw_min = 0.0
            self.yaw_max = 20.0
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
            ] + [0.0] * self.nturbs

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
        # self.layout_dist_initial = np.max(self.x0[0:self.nturbs]) \
        #    - np.min(self.x0[0:self.nturbs])
        self.layout_dist_initial = self._avg_dist(self.x0[0 : self.nturbs])
        # print('initial dist: ', self.layout_dist_initial)

    def plot_layout_opt_results(self):
        """
        This method plots the original and new locations of the turbines in a
        wind farm after layout optimization.
        """
        locsx_old = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in self.x0[0 : self.nturbs]
        ]
        locsy_old = self.fi.layout_y
        locsx = [
            self._unnorm(valx, self.bndx_min, self.bndx_max)
            for valx in self.residual_plant.x[0 : self.nturbs]
        ]
        locsy = self.fi.layout_y

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
