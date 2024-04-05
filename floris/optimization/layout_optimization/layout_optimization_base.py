
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon

from floris import TimeSeries
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)
from floris.wind_data import WindDataBase

from ...logging_manager import LoggingManager


class LayoutOptimization(LoggingManager):
    """
    Base class for layout optimization. This class should not be used directly
    but should be subclassed by a specific optimization method.

    Args:
        fmodel (FlorisModel): A FlorisModel object.
        boundaries (iterable(float, float)): Pairs of x- and y-coordinates
            that represent the boundary's vertices (m).
        min_dist (float, optional): The minimum distance to be maintained
            between turbines during the optimization (m). If not specified,
            initializes to 2 rotor diameters. Defaults to None.
        enable_geometric_yaw (bool, optional): If True, enables geometric yaw
            optimization. Defaults to False.
        use_value (bool, optional): If True, the layout optimization objective
            is to maximize annual value production using the value array in the
            FLORIS model's WindData object. If False, the optimization
            objective is to maximize AEP. Defaults to False.
    """
    def __init__(
        self,
        fmodel,
        boundaries,
        min_dist=None,
        enable_geometric_yaw=False,
        use_value=False,
    ):
        self.fmodel = fmodel.copy() # Does not copy over the wind_data object
        self.fmodel.set(wind_data=fmodel.wind_data)
        self.boundaries = boundaries
        self.enable_geometric_yaw = enable_geometric_yaw
        self.use_value = use_value

        self._boundary_polygon = Polygon(self.boundaries)
        self._boundary_line = LineString(self.boundaries)

        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])

        # If no minimum distance is provided, assume a value of 2 rotor diameters
        if min_dist is None:
            self.min_dist = 2 * self.rotor_diameter
        else:
            self.min_dist = min_dist

        # Check that wind_data is a WindDataBase object
        if (not isinstance(self.fmodel.wind_data, WindDataBase)):
            # NOTE: it is no longer strictly necessary that fmodel use
            # a WindData object, but it is still recommended.
            self.logger.warning(
                "Running layout optimization without a WindData object (e.g. TimeSeries, WindRose, "
                "WindTIRose). We suggest that the user set the wind conditions (and if applicable, "
                "frequencies and values) on the FlorisModel using the wind_data keyword argument "
                "for layout optimizations to capture frequencies and the value of the energy "
                "production accurately. If a WindData object is not defined, uniform frequencies "
                "will be assumed. If use_value is True and a WindData object is not defined, a "
                "value of 1 will be used for each wind condition and layout optimization will "
                "simply be performed to maximize AEP."
            )

        # Establish geometric yaw class
        if self.enable_geometric_yaw:
            self.yaw_opt = YawOptimizationGeometric(
                fmodel,
                minimum_yaw_angle=-30.0,
                maximum_yaw_angle=30.0,
            )
            # TODO: is this being used?
        fmodel.run()

        if self.use_value:
            self.initial_AEP_or_AVP = fmodel.get_farm_AVP()
        else:
            self.initial_AEP_or_AVP = fmodel.get_farm_AEP()

    def __str__(self):
        return "layout"

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    def _get_geoyaw_angles(self):
        # NOTE: requires that child class saves x and y locations
        # as self.x and self.y and updates them during optimization.
        if self.enable_geometric_yaw:
            self.yaw_opt.fmodel_subset.set(layout_x=self.x, layout_y=self.y)
            df_opt = self.yaw_opt.optimize()
            self.yaw_angles = np.vstack(df_opt['yaw_angles_opt'])[:, :]
        else:
            self.yaw_angles = None

        return self.yaw_angles

    # Public methods

    def optimize(self):
        sol = self._optimize()
        return sol

    def plot_layout_opt_results(self):
        x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

        plt.figure(figsize=(9, 6))
        fontsize = 16
        plt.plot(x_initial, y_initial, "ob")
        plt.plot(x_opt, y_opt, "or")
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
        self._nturbs = self.fmodel.core.farm.n_turbines
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fmodel.core.farm.rotor_diameters_sorted[0][0]
