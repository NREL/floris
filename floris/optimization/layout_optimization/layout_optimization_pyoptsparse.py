
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point

from .layout_optimization_base import LayoutOptimization


class LayoutOptimizationPyOptSparse(LayoutOptimization):
    """
    This class provides an interface for optimizing the layout of wind turbines
    using the pyOptSparse optimization library.  The optimization objective is to
    maximize annual energy production (AEP) or annual value production (AVP).

    Args:
        fmodel (FlorisModel): A FlorisModel object.
        boundaries (iterable(float, float)): Pairs of x- and y-coordinates
            that represent the boundary's vertices (m).
        min_dist (float, optional): The minimum distance to be maintained
            between turbines during the optimization (m). If not specified,
            initializes to 2 rotor diameters. Defaults to None.
        solver (str, optional): Sets the solver used by pyOptSparse. Defaults
            to 'SLSQP'.
        optOptions (dict, optional): Dictionary for setting the
            optimization options. Defaults to None.
        timeLimit (float, optional): Variable passed to pyOptSparse optimizer.
            The maximum amount of time for optimizer to run (seconds). If None,
            no time limit is imposed. Defaults to None.
        storeHistory (str, optional): Variable passed to pyOptSparse optimizer.
            File name of the history file into which the history of the
            pyOptSparse optimization will be stored. Defaults to "hist.hist".
        hotStart (str, optional): Variable passed to pyOptSparse optimizer.
            File name of the history file to “replay” for the optimization.
            If None, pyOptSparse initializes the optimization from scratch.
            Defaults to None.
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
        solver=None,
        optOptions=None,
        timeLimit=None,
        storeHistory='hist.hist',
        hotStart=None,
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

        self.x0 = self._norm(self.fmodel.layout_x, self.xmin, self.xmax)
        self.y0 = self._norm(self.fmodel.layout_y, self.ymin, self.ymax)

        self.storeHistory = storeHistory
        self.timeLimit = timeLimit
        self.hotStart = hotStart
        self.enable_geometric_yaw = enable_geometric_yaw

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

        # Instantiate pyOptSparse optimization object with name and objective function
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

        # Compute turbine yaw angles using PJ's geometric code (if enabled)
        yaw_angles = self._get_geoyaw_angles()
        # Update turbine map with turbine locations and yaw angles
        self.fmodel.set(layout_x=self.x, layout_y=self.y, yaw_angles=yaw_angles)
        self.fmodel.run()

        # Compute the objective function
        funcs = {}
        if self.use_value:
            funcs["obj"] = -1 * self.fmodel.get_farm_AVP() / self.initial_AEP_or_AVP
        else:
            funcs["obj"] = -1 * self.fmodel.get_farm_AEP() / self.initial_AEP_or_AVP

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
            "x", self.nturbs, varType="c", lower=0.0, upper=1.0, value=self.x0
        )
        optProb.addVarGroup(
            "y", self.nturbs, varType="c", lower=0.0, upper=1.0, value=self.y0
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
            boundary_con[i] = loc.distance(self._boundary_line)
            if self._boundary_polygon.contains(loc) is True:
                boundary_con[i] *= -1.0

        return boundary_con

    def _get_initial_and_final_locs(self):
        x_initial = self._unnorm(self.x0, self.xmin, self.xmax)
        y_initial = self._unnorm(self.y0, self.ymin, self.ymax)
        x_opt, y_opt = self.get_optimized_locs()
        return x_initial, y_initial, x_opt, y_opt

    def get_optimized_locs(self):
        x_opt = self._unnorm(self.sol.getDVs()["x"], self.xmin, self.xmax)
        y_opt = self._unnorm(self.sol.getDVs()["y"], self.ymin, self.ymax)
        return x_opt, y_opt
