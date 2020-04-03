# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from .layout import LayoutOptimization
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


class PowerDensityOptimization(LayoutOptimization):
    def __init__(self, fi, boundaries,
                           wd,
                           ws,
                           freq,
                           AEP_initial,
                           yawbnds=None,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method='SLSQP',
                           opt_options=None):

        super().__init__(fi, boundaries,
                         wd,
                         ws,
                         freq,
                         AEP_initial,
                         x0=x0,
                         bnds=bnds,
                         min_dist=min_dist,
                         opt_method=opt_method,
                         opt_options=opt_options)
        self.epsilon = np.finfo(float).eps
        self.counter = 0

        if opt_options is None:
            self.opt_options = {'maxiter': 100, 'disp': True, \
                         'iprint': 2, 'ftol': 1e-9}

    def _generate_constraints(self):
        # grad_constraint1 = grad(self._space_constraint)
        # grad_constraint2 = grad(self._distance_from_boundaries)

        tmp1 = {'type': 'ineq','fun' : lambda x,*args: \
                self._space_constraint(x, self.min_dist), \
                'args':(self.min_dist,)}
        tmp2 = {'type': 'ineq','fun' : lambda x,*args: \
                self._distance_from_boundaries(x, self.boundaries_norm), \
                'args':(self.boundaries_norm,)}
        tmp3 = {'type': 'ineq','fun' : lambda x,*args: \
                self._AEP_constraint(x)}

        self.cons = [tmp1, tmp2, tmp3]

    def _set_opt_bounds(self):
        self.bnds = [(0.0, 1.0) for _ in range(2*self.nturbs + self.nturbs*len(self.wd))]

    def _change_coordinates(self, locsx, locsy):
        # Parse the layout coordinates
        layout_array = [locsx, locsy]

        # Update the turbine map in floris
        self.fi.reinitialize_flow_field(layout_array=layout_array)

    def _powDens_opt(self, optVars):
        locsx = optVars[0:self.nturbs]
        locsy = optVars[self.nturbs:2*self.nturbs]

        locsx_unnorm = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                        for valx in locsx]
        locsy_unnorm = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                        for valy in locsy]

        turb_controls = [optVars[2*self.nturbs + i*self.nturbs:3*self.nturbs + i*self.nturbs] for i in range(len(self.wd))]

        turb_controls_unnorm = [self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in turb_controls]

        self._change_coordinates(locsx_unnorm, locsy_unnorm)
        opt_area = self.find_layout_area(locsx_unnorm + locsy_unnorm)

        AEP_sum = 0.

        for i in range(len(self.wd)):
            for j, turbine in enumerate(self.fi.floris.farm.turbine_map.turbines):
                turbine.yaw_angle = turb_controls_unnorm[i][j]

            AEP_sum = AEP_sum + self._AEP_single_wd(self.wd[i], 
                                                    self.ws[i], 
                                                    self.freq[i])

        # print('AEP ratio: ', AEP_sum/self.AEP_initial)

        return -1*AEP_sum/self.AEP_initial*self.initial_area/opt_area

    def _AEP_constraint(self, optVars):
        locsx = optVars[0:self.nturbs]
        locsy = optVars[self.nturbs:2*self.nturbs]

        locsx_unnorm = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                        for valx in locsx]
        locsy_unnorm = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                        for valy in locsy]

        turb_controls = [optVars[2*self.nturbs + i*self.nturbs:3*self.nturbs + i*self.nturbs] for i in range(len(self.wd))]

        turb_controls_unnorm = [self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in turb_controls]

        self._change_coordinates(locsx_unnorm, locsy_unnorm)

        AEP_sum = 0.

        for i in range(len(self.wd)):
            for j, turbine in enumerate(self.fi.floris.farm.turbine_map.turbines):
                turbine.yaw_angle = turb_controls_unnorm[i][j]

            AEP_sum = AEP_sum + self._AEP_single_wd(self.wd[i], 
                                                    self.ws[i], 
                                                    self.freq[i])

        return (AEP_sum/self.AEP_initial - 1.)

    def _optimize(self):
        self.residual_plant = minimize(self._powDens_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                constraints=self.cons,
                                options=self.opt_options)

        opt_results = self.residual_plant.x
        
        return opt_results

    def optimize(self):
        # TODO: update docs
        """
        Find optimized layout of wind turbines for power production given
        fixed atmospheric conditions (wind speed, direction, etc.).
        
        Returns:
            opt_locs (iterable): optimized locations of each turbine.
        """
        print('=====================================================')
        print('Optimizing turbine layout...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_locs_norm = self._optimize()

        print('Optimization complete!')

        opt_locs = [[self._unnorm(valx, self.bndx_min, self.bndx_max) \
            for valx in opt_locs_norm[0:self.nturbs]], \
            [self._unnorm(valy, self.bndy_min, self.bndy_max) \
            for valy in opt_locs_norm[self.nturbs:2*self.nturbs]]]

        return opt_locs

    def reinitialize_opt(self, boundaries=None,
                           yawbnds=None,
                           wd=None,
                           ws=None,
                           freq=None,
                           AEP_initial=None,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method=None,
                           opt_options=None):
        """
        Reintializes parameter values for the optimization.
        
        This method reinitializes the optimization parameters and 
        bounds to the supplied values or uses what is currently stored.
        
        Args:
            boundaries (iterable): A list of pairs of floats that 
                represent the boundary's vertices. Defaults to None.
            wd (np.array, optional): An array of wind directions. 
                Defaults to None.
            ws (np.array, optional): An array of wind speeds. Defaults 
                to None.
            freq (np.array, optional): An array of wind direction 
                frequency values. Defaults to None.
            AEP_initial (float, optional): Initial Annual Energy 
                Production used in normalizing optimization. Defaults 
                to None. Initializes to the AEP of the current FLORIS 
                object.
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]). Defaults to 
                None. Initializes to the current turbine locations.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to the min. and max. 
                values of the boundaries iterable.
            min_dist (float, optional): The minimum distance to be 
                maitained between turbines during the optimization. 
                Defaults to None. Initializes to 2 rotor diameters.
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dict, optional): Dicitonary for setting the 
                optimization options. Defaults to None.
        """
        if boundaries is not None:
            self.boundaries = boundaries
            self.bndx_min = np.min([val[0] for val in boundaries])
            self.bndy_min = np.min([val[1] for val in boundaries])
            self.bndx_max = np.max([val[0] for val in boundaries])
            self.bndy_max = np.max([val[1] for val in boundaries])
            self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
                                  self.bndx_max), self._norm(val[1], \
                                  self.bndy_min, self.bndy_max)] \
                                  for val in self.boundaries]
        if yawbnds is not None:
            self.yaw_min = yawbnds[0]
            self.yaw_max = yawbnds[1]
        else:
            self.yaw_min = 0.0
            self.yaw_max = 25.0
        if wd is not None:
            self.wd = wd
        if ws is not None:
            self.ws = ws
        if freq is not None:
            self.freq = freq
        if AEP_initial is not None:
            self.AEP_initial = AEP_initial
        else:
            self.AEP_initial = self.fi.get_farm_AEP(self.wd, \
                                                    self.ws, self.freq)
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [self._norm(coord.x1, self.bndx_min, self.bndx_max) for \
                coord in self.fi.floris.farm.turbine_map.coords] \
                + [self._norm(coord.x2, self.bndy_min, self.bndy_max) for \
                coord in self.fi.floris.farm.turbine_map.coords] \
                + [self._norm(5.0, self.yaw_min, self.yaw_max)]*len(self.wd)*self.nturbs
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 4*self.fi.floris.farm.turbines[0].rotor_diameter
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options

        self.layout_x_orig = [coord.x1 for coord in \
                              self.fi.floris.farm.turbine_map.coords] 
        self.layout_y_orig = [coord.x2 for coord in \
                              self.fi.floris.farm.turbine_map.coords]

        self._generate_constraints()

        self.initial_area = self.find_layout_area(
                                    self.layout_x_orig + self.layout_y_orig)

    def find_layout_area(self, locs):
        locsx = locs[0:self.nturbs]
        locsy = locs[self.nturbs:]

        points = zip(locsx, locsy)
        points = np.array(list(points))

        hull = self.convex_hull(points)

        area = self.polygon_area(np.array([val[0] for val in hull]), np.array([val[1] for val in hull]))

        return area

    def convex_hull(self, points):
        # find two hull points, U, V, and split to left and right search
        u = min(points, key=lambda p: p[0])
        v = max(points, key=lambda p: p[0])
        left, right = self.split(u, v, points), self.split(v, u, points)

        # find convex hull on each side
        return [v] + self.extend(u, v, left) + [u] + self.extend(v, u, right) + [v]

    def polygon_area(self, x, y):
        # coordinate shift
        x_ = x - x.mean()
        y_ = y - y.mean()
        
        correction = x_[-1] * y_[0] - y_[-1]* x_[0]
        main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
        return 0.5*np.abs(main_area + correction)

    def split(self, u, v, points):
        # return points on left side of UV
        return [p for p in points if np.cross(p - u, v - u) < 0]

    def extend(self, u, v, points):
        if not points:
            return []

        # find furthest point W, and split search to WV, UW
        w = min(points, key=lambda p: np.cross(p - u, v - u))
        p1, p2 = self.split(w, v, points), self.split(u, w, points)
        return self.extend(w, v, p1) + [w] + self.extend(u, w, p2)

    def plot_opt_results(self):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx_old = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                     for valx in self.x0[0:self.nturbs]]
        locsy_old = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                     for valy in self.x0[self.nturbs:2*self.nturbs]]
        locsx = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                 for valx in self.residual_plant.x[0:self.nturbs]]
        locsy = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                 for valy in self.residual_plant.x[self.nturbs:2*self.nturbs]]

        plt.figure(figsize=(9,6))
        fontsize= 16
        plt.plot(locsx_old, locsy_old, 'ob')
        plt.plot(locsx, locsy, 'or')
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        plt.legend(['Old locations', 'New locations'], loc='lower center', \
            bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fontsize)

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts)-1:
                plt.plot([verts[i][0], verts[0][0]], \
                         [verts[i][1], verts[0][1]], 'b')        
            else:
                plt.plot([verts[i][0], verts[i+1][0]], \
                         [verts[i][1], verts[i+1][1]], 'b')