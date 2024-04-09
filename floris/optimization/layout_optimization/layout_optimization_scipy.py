
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Point

from .layout_optimization_base import LayoutOptimization


class LayoutOptimizationScipy(LayoutOptimization):
    """
    This class provides an interface for optimizing the layout of wind turbines
    using the Scipy optimization library.  The optimization objective is to
    maximize annual energy production (AEP) or annual value production (AVP).


    Args:
        fmodel (FlorisModel): A FlorisModel object.
        boundaries (iterable(float, float)): Pairs of x- and y-coordinates
            that represent the boundary's vertices (m).
        bnds (iterable, optional): Bounds for the optimization
            variables (pairs of min/max values for each variable (m)). If
            none are specified, they are set to 0 and 1. Defaults to None.
        min_dist (float, optional): The minimum distance to be maintained
            between turbines during the optimization (m). If not specified,
            initializes to 2 rotor diameters. Defaults to None.
        solver (str, optional): Sets the solver used by Scipy. Defaults to 'SLSQP'.
        optOptions (dict, optional): Dictionary for setting the
            optimization options. Defaults to None.
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
        bnds=None,
        min_dist=None,
        solver='SLSQP',
        optOptions=None,
        enable_geometric_yaw=False,
        use_value=False,
    ):

        super().__init__(
            fmodel,
            boundaries,
            min_dist=min_dist,
            enable_geometric_yaw=enable_geometric_yaw,
            use_value=use_value
        )

        self.boundaries_norm = [
            [
                self._norm(val[0], self.xmin, self.xmax),
                self._norm(val[1], self.ymin, self.ymax),
            ]
            for val in self.boundaries
        ]
        self.x0 = [
            self._norm(x, self.xmin, self.xmax)
            for x in self.fmodel.layout_x
        ] + [
            self._norm(y, self.ymin, self.ymax)
            for y in self.fmodel.layout_y
        ]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if solver is not None:
            self.solver = solver

        default_optOptions = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-9, "eps":0.01}
        if optOptions is not None:
            self.optOptions = {**default_optOptions, **optOptions}
        else:
            self.optOptions = default_optOptions

        self._generate_constraints()


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
            self._unnorm(valx, self.xmin, self.xmax)
            for valx in locs[0 : self.nturbs]
        ] + [
            self._unnorm(valy, self.ymin, self.ymax)
            for valy in locs[self.nturbs : 2 * self.nturbs]
        ]
        self._change_coordinates(locs_unnorm)
        # Compute turbine yaw angles using PJ's geometric code (if enabled)
        yaw_angles = self._get_geoyaw_angles()
        self.fmodel.set_operation(yaw_angles=yaw_angles)
        self.fmodel.run()

        if self.use_value:
            return -1 * self.fmodel.get_farm_AVP() / self.initial_AEP_or_AVP
        else:
            return -1 * self.fmodel.get_farm_AEP() / self.initial_AEP_or_AVP


    def _change_coordinates(self, locs):
        # Parse the layout coordinates
        layout_x = locs[0 : self.nturbs]
        layout_y = locs[self.nturbs : 2 * self.nturbs]

        # Store on object for use in geoyaw code
        self.x = layout_x
        self.y = layout_y

        # Update the turbine map in floris
        self.fmodel.set(layout_x=layout_x, layout_y=layout_y)

    def _generate_constraints(self):
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

    def _space_constraint(self, x_in, rho=500):
        x = [
            self._unnorm(valx, self.xmin, self.xmax)
            for valx in x_in[0 : self.nturbs]
        ]
        y =  [
            self._unnorm(valy, self.ymin, self.ymax)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]

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
        x = [
            self._unnorm(valx, self.xmin, self.xmax)
            for valx in x_in[0 : self.nturbs]
        ]
        y =  [
            self._unnorm(valy, self.ymin, self.ymax)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]
        boundary_con = np.zeros(self.nturbs)
        for i in range(self.nturbs):
            loc = Point(x[i], y[i])
            boundary_con[i] = loc.distance(self._boundary_line)
            if self._boundary_polygon.contains(loc) is True:
                boundary_con[i] *= 1.0
            else:
                boundary_con[i] *= -1.0

        return boundary_con

    def _get_initial_and_final_locs(self):
        x_initial = [
            self._unnorm(valx, self.xmin, self.xmax)
            for valx in self.x0[0 : self.nturbs]
        ]
        y_initial = [
            self._unnorm(valy, self.ymin, self.ymax)
            for valy in self.x0[self.nturbs : 2 * self.nturbs]
        ]
        x_opt = [
            self._unnorm(valx, self.xmin, self.xmax)
            for valx in self.residual_plant.x[0 : self.nturbs]
        ]
        y_opt = [
            self._unnorm(valy, self.ymin, self.ymax)
            for valy in self.residual_plant.x[self.nturbs : 2 * self.nturbs]
        ]
        return x_initial, y_initial, x_opt, y_opt


    # Public methods

    def optimize(self):
        """
        This method finds the optimized layout of wind turbines for power
        production given the provided frequencies of occurrence of wind
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
                self._unnorm(valx, self.xmin, self.xmax)
                for valx in opt_locs_norm[0 : self.nturbs]
            ],
            [
                self._unnorm(valy, self.ymin, self.ymax)
                for valy in opt_locs_norm[self.nturbs : 2 * self.nturbs]
            ],
        ]

        return opt_locs
