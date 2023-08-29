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
from shapely.geometry import Polygon, MultiPolygon

from floris.tools.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)

from ....logging_manager import LoggerBase


class LayoutOptimization(LoggerBase):
    def __init__(self, fi, boundaries, min_dist=None, freq=None, enable_geometric_yaw=False):
        self.fi = fi.copy()

        self.enable_geometric_yaw = enable_geometric_yaw

        # Allow boundaries to be set either as a list of corners or as a 
        # nested list of corners (for seperable regions)
        self.boundaries = boundaries
        b_depth = list_depth(boundaries)

        boundary_specification_error_msg = (
            "boundaries should be a list of coordinates (specifed as (x,y) "+\
            "tuples) or as a list of list of tuples (for seperable regions)."
        )
        
        if b_depth == 1:
            self._boundary_polygon = MultiPolygon([Polygon(self.boundaries)])
            self._boundary_line = self._boundary_polygon.boundary
        elif b_depth == 2:
            if not isinstance(self.boundaries[0][0], tuple):
                raise TypeError(boundary_specification_error_msg)
            self._boundary_polygon = MultiPolygon([Polygon(p) for p in self.boundaries])
            self._boundary_line = self._boundary_polygon.boundary
        else:
            raise TypeError(boundary_specification_error_msg) 

        self.xmin, self.ymin, self.xmax, self.ymax = self._boundary_polygon.bounds

        # If no minimum distance is provided, assume a value of 2 rotor diamters
        if min_dist is None:
            self.min_dist = 2 * self.rotor_diameter
        else:
            self.min_dist = min_dist

        # If freq is not provided, give equal weight to all wind conditions
        if freq is None:
            self.freq = np.ones((
                self.fi.floris.flow_field.n_wind_directions,
                self.fi.floris.flow_field.n_wind_speeds
            ))
            self.freq = self.freq / self.freq.sum()
        else:
            self.freq = freq

        # Establish geometric yaw class
        if self.enable_geometric_yaw:
            self.yaw_opt = YawOptimizationGeometric(
                fi,
                minimum_yaw_angle=-30.0,
                maximum_yaw_angle=30.0,
                exploit_layout_symmetry=False
            )

        self.initial_AEP = fi.get_farm_AEP(self.freq)

    def __str__(self):
        return "layout"

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    def _get_geoyaw_angles(self):
        if self.enable_geometric_yaw:
            df_opt = self.yaw_opt.optimize()
            self.yaw_angles = np.vstack(df_opt['yaw_angles_opt'])[:, None, :]
        else:
            self.yaw_angles = None

        return self.yaw_angles

    # Public methods

    def optimize(self):
        sol = self._optimize()
        return sol

    def plot_layout_opt_results(self, plot_boundary_dict={}, ax=None, fontsize=16):
        
        x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

        # Generate axis, if needed
        if ax is None:
            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
        
        default_plot_boundary_dict = {
            "color":"None",
            "alpha":1,
            "edgecolor":"b",
            "linewidth":2
        }
        plot_boundary_dict = {**default_plot_boundary_dict, **plot_boundary_dict}

        self.plot_layout_opt_boundary(plot_boundary_dict, ax=ax)
        ax.plot(x_initial, y_initial, "ob", label="Initial locations")
        ax.plot(x_opt, y_opt, "or", label="New locations")
        ax.set_xlabel("x (m)", fontsize=fontsize)
        ax.set_ylabel("y (m)", fontsize=fontsize)
        ax.grid(True)
        ax.tick_params(which="both", labelsize=fontsize)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )

        return ax

    def plot_layout_opt_boundary(self, plot_boundary_dict={}, ax=None):

        # Generate axis, if needed
        if ax is None:
            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")

        default_plot_boundary_dict = {
            "color":"k",
            "alpha":0.1,
            "edgecolor":None
        }

        plot_boundary_dict = {**default_plot_boundary_dict, **plot_boundary_dict}

        for line in self._boundary_line.geoms:
            xy = np.array(line.coords)
            ax.fill(xy[:,0], xy[:,1], **plot_boundary_dict)
        ax.grid(True)

        return ax

    def plot_progress(self, ax=None):

        if not hasattr(self, "aep_candidate_log"):
            raise NotImplementedError(
                "plot_progress not yet configured for "+self.__class__.__name__
            )

        if ax is None:
            fig, ax = plt.subplots(1,1)

        aep_log_array = np.array(self.aep_candidate_log)
        
        if len(aep_log_array.shape) == 1: # Just one AEP candidate per step
            ax.plot(np.arange(len(aep_log_array)), aep_log_array, color="k")
        elif len(aep_log_array.shape) == 2: # Multiple AEP candidates per step
            for i in range(aep_log_array.shape[1]):
                ax.plot(np.arange(len(aep_log_array)), aep_log_array[:,i], 
                    color="lightgray")

        # Plot aesthetics
        ax.grid(True)
        ax.set_xlabel("Optimization step [-]")
        ax.set_ylabel("AEP value [GWh]") 
        # TODO: Check optimizers for AEP unit consistency

        return ax


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
        return self.fi.floris.farm.rotor_diameters_sorted[0][0][0]

# Helper functions

def list_depth(l):
    if isinstance(l, list) and len(l) > 0:
        return 1 + max(list_depth(item) for item in l)
    else:
        return 0