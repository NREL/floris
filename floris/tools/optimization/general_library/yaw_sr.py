import numpy as np
from scipy.optimize import minimize

from floris.tools.optimization.general_library.optimization import YawOptimization


class YawOptimizationSR(YawOptimization):
    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        opt_options=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        exclude_downstream_turbines=False,
    ):
        """
            Args:
            opt_options (dictionary, optional): Optimization options used by
                scipy.optimize.minize. If none are specified, they are set to
                {'maxiter': 100, 'disp': False, 'iprint': 1, 'ftol': 1e-7,
                'eps': 0.01}. Defaults to None.
        """
        if opt_options is None:
            # Default SR parameters
            opt_options={
                        "Ny_passes": [5, 5],
                        "refine_solution": True,
                        "refine_method": "SLSQP",
                        "refine_options": {
                            'maxiter': 10,
                            'disp': True,
                            'iprint': 1,
                            'ftol': 1e-7,
                            'eps': 0.01
                        }
            }
        else:
            # Confirm Ny_firstpass and Ny_secondpass are odd integers
            for Ny in opt_options["Ny_passes"]:
                if (not isinstance(Ny, int)) or (Ny % 2 == 0):
                    raise ValueError("Ny_passes must contain exclusively odd integers")
        
        super().__init__(
            fi=fi,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            bnds=bnds,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            turbine_weights=turbine_weights,
            calc_init_power=False,
            exclude_downstream_turbines=exclude_downstream_turbines,
            )

        self.opt_options = opt_options
        self._init_solution_refiner()

    def _init_solution_refiner(self):
        if self.opt_options["refine_solution"]:
            from floris.tools.optimization.general_library import yaw_scipy
            self.refinement_solver = yaw_scipy.YawOptimizationScipy(
                fi=self.fi,
                minimum_yaw_angle=self.minimum_yaw_angle,
                maximum_yaw_angle=self.maximum_yaw_angle,
                yaw_angles_baseline=self.yaw_angles_baseline,
                x0=None,
                bnds=self.bnds,
                opt_method=self.opt_options["refine_method"],
                opt_options=self.opt_options["refine_options"],
                include_unc=self.include_unc,
                unc_pmfs=self.unc_pmfs,
                unc_options=self.unc_options,
                turbine_weights=self.turbine_weights,
                exclude_downstream_turbines=self.exclude_downstream_turbines
            )

    def _serial_refine_single_pass(self, yaw_grid):
        # Get a list of the turbines in order of x and sort front to back
        layout_x = self.fi.layout_x
        layout_y = self.fi.layout_y
        wind_direction = self.fi.floris.farm.wind_direction[0]
        layout_x_rot = (
            np.cos((wind_direction - 270.0) * np.pi / 180.0) * layout_x
            - np.sin((wind_direction - 270.0) * np.pi / 180.0) * layout_y
        )
        turbines_ordered = np.argsort(layout_x_rot)

        # Remove turbines that need not be optimized
        turbines_ordered = [ti for ti in turbines_ordered if ti in self.turbs_to_opt]

        J_farm_opt = -1  # Initialize optimal solution
        yaw_angles_opt = np.array(self.yaw_angles_template, dtype=float)
        for ti in turbines_ordered:
            yaw_opt = self.yaw_angles_template[ti]  # Initialize
            for yaw in yaw_grid[ti]:
                if (not (yaw == self.yaw_angles_template[ti]) & ti > 0):  # Exclude case that we already evaluated
                    yaw_angles_opt[ti] = yaw
                    self.fi.calculate_wake(yaw_angles=yaw_angles_opt)
                    turbine_powers = self.fi.get_turbine_power(
                        include_unc=self.include_unc,
                        unc_pmfs=self.unc_pmfs,
                        unc_options=self.unc_options,
                    )
                    test_power = np.dot(self.turbine_weights, turbine_powers)
                    if test_power > J_farm_opt:
                        J_farm_opt = test_power
                        yaw_opt = yaw
            yaw_angles_opt[ti] = yaw_opt

        return yaw_angles_opt

    def _refine_solution(self, x0):
        self.refinement_solver.x0 = x0  # Overwrite initial condition
        return self.refinement_solver.optimize()
        
    def optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditions (wind speed, direction, etc.)
        using the scipy.optimize.minimize function.

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        # Reduce degrees of freedom and check if optimization necessary
        self._reduce_control_variables()
        if len(self.turbs_to_opt) <= 0:
            return self.yaw_angles_template

        # Initialization
        yaw_search_space = [[]] * self.nturbs  # Initialize as empty
        yaw_grid_offsets = [np.mean(b) for b in self.bnds]
        bounds_relative = [[b[0]-y, b[1]-y] for b, y in zip(self.bnds, yaw_grid_offsets)]

        # Perform each pass with Ny yaw angles
        for ii, Ny in enumerate(self.opt_options["Ny_passes"]):
            for ti in range(self.nturbs):
                lb = bounds_relative[ti][0]
                if (lb + yaw_grid_offsets[ti]) < self.bnds[ti][0]:
                    lb = (self.bnds[ti][0] - yaw_grid_offsets[ti])
        
                ub = bounds_relative[ti][1]
                if (ub + yaw_grid_offsets[ti]) > self.bnds[ti][1]:
                    ub = (self.bnds[ti][1] - yaw_grid_offsets[ti])

                yaw_search_space[ti] = (
                    yaw_grid_offsets[ti] + np.linspace(lb, ub, Ny)
                )

            # Single pass through all turbines
            yaw_angles_opt = self._serial_refine_single_pass(yaw_search_space)

            # Update variables if we are to do more passes
            if (ii < len(self.opt_options["Ny_passes"]) - 1):
                yaw_angles_template = yaw_angles_opt  # Overwrite initial cond.
                yaw_grid_offsets = yaw_angles_opt  # Refine search space
                for ti in range(self.nturbs):
                    dx = float(np.diff(bounds_relative[ti]) / (Ny - 1))
                    bounds_relative[ti] = [-dx/2.0, dx/2.0]

        if self.opt_options["refine_solution"]:
            yaw_angles_opt = self._refine_solution(x0=yaw_angles_opt)

        return yaw_angles_opt
