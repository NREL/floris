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

from .layout import LayoutOptimization
from .base_COE import BaseCOE
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


class LayoutHeightOptimization(LayoutOptimization):
    """
    LayoutHeightOptimization is a subclass of 
    :py:class:`~.tools.optimization.scipy.layout.LayoutOptimization` that
    performs layout and turbine height optimization. This optimization method
    aims to minimize Cost of Energy (COE) by changing individual turbine
    locations and all turbine heights across the wind farm. Note that the
    changing turbine height applies to all turbines, i.e. although the turbine
    height is changing, all turbines will be assigned the same turbine height.
    """
    def __init__(self, fi, boundaries,
                           height_lims,
                           wd,
                           ws,
                           freq,
                           AEP_initial,
                           COE_initial,
                           plant_kw,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method='SLSQP',
                           opt_options=None):
        """
        Instantiate LayoutHeightOptimization object with a FlorisInterface
        object and assign parameter values.
        
        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`): 
                Interface used to interact with the Floris object.
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            height_lims (iterable): A list of the minimum and maximum 
                height limits for the optimization (m). Each value only 
                needs to be defined once since all the turbine heights 
                are the same (ie. [h_min, h_max]).
            wd (np.array): An array of wind directions (deg).
            ws (np.array): An array of wind speeds (m/s).
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values.
            AEP_initial (float): The initial Annual Energy 
                Production used for normalization in the optimization (Wh)
                (TODO: Is Watt-hours the correct unit?).
            COE_initial (float): Initial Cost of Energy used for 
                normalization in the optimization ($/kWh).
            plant_kw (float): The rating of the entire wind plant (kW). 
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]), and the 
                initial turbine hub height (m). If none are provided, x0
                initializes to the current turbine locations and hub height.
                Defaults to None. 
            bnds (iterable, optional): Bounds for the optimization 
                variables (TODO: just coordinates, or height too?) (pairs of
                min/max values for each variable (m)). If none are specified,
                they are set to the min. and max. of the boundaries iterable.
                Defaults to None.
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
        super().__init__(fi, boundaries, wd, ws, freq, AEP_initial)
        self.epsilon = np.finfo(float).eps

        self.COE_model = BaseCOE(self)
        
        self.reinitialize_opt_height(
            boundaries=boundaries,
            height_lims=height_lims,
            wd = wd,
            ws = ws,
            freq = freq,
            AEP_initial = AEP_initial,
            COE_initial=COE_initial,
            plant_kw=plant_kw,
            x0=x0,
            bnds=bnds,
            min_dist=min_dist,
            opt_method=opt_method,
            opt_options=opt_options
        )

    # Private methods

    def _fCp_outside(self):
        pass # for future use

    def _fCt_outside(self):
        pass # for future use

    def _set_initial_conditions(self):
        self.x0.append( \
            self._norm(self.fi.floris.farm.turbines[0].hub_height, \
                        self.bndh_min, self.bndh_max))
    
    def _set_opt_bounds_height(self):
        self.bnds.append((0., 1.))

    def _optimize(self):
        self.residual_plant = minimize(self._COE_layout_height_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                constraints=self.cons,
                                options=self.opt_options)

        opt_results = self.residual_plant.x
        
        return opt_results
    
    def _COE_layout_height_opt(self, opt_vars):
        locs = self._unnorm(
            opt_vars[0:2*self.nturbs],
            self.bndx_min,
            self.bndx_max
        )
        height = self._unnorm(opt_vars[-1], self.bndh_min, self.bndh_max)

        self._change_height(height)
        self._change_coordinates(locs)
        AEP_sum = self._AEP_loop_wd()
        COE = self.COE_model.COE(height, AEP_sum)

        return COE/self.COE_initial

    def _change_height(self, height):
        if isinstance(height, float) or isinstance(height, int):
            for turb in self.fi.floris.farm.turbines:
                turb.hub_height = height
        else:
            for k, turb in enumerate(self.fi.floris.farm.turbines):
                turb.hub_height = height[k]

        self.fi.reinitialize_flow_field(layout_array=((self.fi.layout_x, \
                                                       self.fi.layout_y)))

    # Public methods

    def reinitialize_opt_height(self, boundaries=None,
                           height_lims=None,
                           wd = None,
                           ws = None,
                           freq = None,
                           AEP_initial=None,
                           COE_initial = None,
                           plant_kw=None,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method=None,
                           opt_options=None):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.
        
        Args:
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            height_lims (iterable): A list of the minimum and maximum 
                height limits for the optimization (m). Each value only 
                needs to be defined once since all the turbine heights 
                are the same (ie. [h_min, h_max]). Defaults to None.
            wd (np.array): An array of wind directions (deg). Defaults to None.
            ws (np.array): An array of wind speeds (m/s). Defaults to None.
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values. Defaults to None.
            AEP_initial (float): The initial Annual Energy 
                Production used for normalization in the optimization (Wh).
                Defaults to None.
            COE_initial (float): Initial Cost of Energy used for 
                normalization in the optimization ($/kWh). Defaults to None.
            plant_kw (float): The rating of the entire wind plant (kW).
                Defaults to None.
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]), and the 
                initial turbine hub height (m). If none are provided, x0
                initializes to the current turbine locations and hub height.
                Defaults to None. 
            bnds (iterable, optional): Bounds for the optimization 
                variables (TODO: just coordinates, or height too?) (pairs of
                min/max values for each variable (m)). If none are specified,
                they are set to the min. and max. of the boundaries iterable.
                Defaults to None.
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
        LayoutOptimization.reinitialize_opt(self, boundaries=boundaries,
                           wd=wd,
                           ws=ws,
                           freq=freq,
                           AEP_initial=AEP_initial,
                           x0=x0,
                           bnds=bnds,
                           min_dist=min_dist,
                           opt_method=opt_method,
                           opt_options=opt_options)

        if height_lims is not None:
            self.bndh_min = height_lims[0]
            self.bndh_max = height_lims[1]
        if COE_initial is not None:
            self.COE_initial = COE_initial
        if plant_kw is not None:
            self.plant_kw = plant_kw

        self._set_initial_conditions()
        self._set_opt_bounds_height()

    def optimize(self):
        """
        This method finds the optimized layout of wind turbines and wind
        turbine height for power production and cost of energy given the
        provided frequencies of occurance of wind conditions (wind speed,
        direction).
        
        Returns:
            (iterable): A list containing the optimized (x, y) locations of
            each turbine followed by the optimized height for all turbines (m).
        """
        print('=====================================================')
        print('Optimizing turbine layout and height...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_results_norm = self._optimize()

        print('Optimization complete!')

        opt_locs = [[self._unnorm(valx, self.bndx_min, self.bndx_max) \
            for valx in opt_results_norm[0:self.nturbs]], \
            [self._unnorm(valy, self.bndy_min, self.bndy_max) \
            for valy in opt_results_norm[self.nturbs:2*self.nturbs]]]

        opt_height = [self._unnorm(opt_results_norm[-1], \
                      self.bndh_min, self.bndh_max)]

        return [opt_locs, opt_height]

    def get_farm_COE(self):
        """
        This method returns the cost of energy (COE) for the wind farm.

        Returns:
            float: The cost of energy for a wind plant in units of $/kWh.
        """
        AEP_sum = self._AEP_loop_wd()
        height = self.fi.floris.farm.turbines[0].hub_height
        return self.COE_model.COE(height, AEP_sum)
