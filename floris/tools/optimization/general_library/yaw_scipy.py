import numpy as np
from scipy.optimize import minimize

from floris.tools.optimization.general_library.optimization import YawOptimization


class YawOptimizationScipy(YawOptimization):
    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        opt_method="SLSQP",
        opt_options=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        exclude_downstream_turbines=False,
    ):
        """
            Args:
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to 'SLSQP'.
            opt_options (dictionary, optional): Optimization options used by
                scipy.optimize.minize. If none are specified, they are set to
                {'maxiter': 100, 'disp': False, 'iprint': 1, 'ftol': 1e-7,
                'eps': 0.01}. Defaults to None.
        """
        if opt_options is None:
            # Default SciPy parameters
            opt_options = {
                "maxiter": 50,
                "disp": True,
                "iprint": 2,
                "ftol": 1e-12,
                "eps": 0.1,
            }
        
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
            calc_init_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            )

        self.opt_method = opt_method
        self.opt_options = opt_options

    def _cost_full_yaw_angle_array(self, yaw_angles_normalized):
        # Undo normalization in yaw_angles
        yaw_angles = self._unnorm(
            yaw_angles_normalized,
            self.minimum_yaw_angle,
            self.maximum_yaw_angle
        )

        # Calculate cost
        self.fi.calculate_wake(yaw_angles=yaw_angles)
        turbine_powers = self.fi.get_turbine_power(
            include_unc=self.include_unc,
            unc_pmfs=self.unc_pmfs,
            unc_options=self.unc_options,
        )
        
        # Normalize cost
        J = -1.0 * np.dot(self.turbine_weights, turbine_powers)
        J_norm = J / self.initial_farm_power

        return J_norm

    def _cost(self, yaw_angles_subset_norm):
        # Combine template yaw angles and subset
        yaw_angles_norm = self._norm(
            self.yaw_angles_template,
            self.minimum_yaw_angle,
            self.maximum_yaw_angle
        )
        yaw_angles_norm[self.turbs_to_opt] = yaw_angles_subset_norm
        return self._cost_full_yaw_angle_array(yaw_angles_norm)

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

        # Initialize full optimal yaw angle array
        opt_yaw_angles = np.array(self.yaw_angles_template, copy=True)

        # Reduce number of variables and normalize yaw angles to [0, 1]
        self._normalize_control_variables()

        # Use SciPy to find the optimal solutions
        self.residual_plant = minimize(
            self._cost,
            self.x0_norm,
            method=self.opt_method,
            bounds=self.bnds_norm,
            options=self.opt_options,
        )
        opt_yaw_angles_subset = self._unnorm(
            self.residual_plant.x, self.minimum_yaw_angle, self.maximum_yaw_angle
        )
        opt_yaw_angles[self.turbs_to_opt] = opt_yaw_angles_subset
        return opt_yaw_angles
