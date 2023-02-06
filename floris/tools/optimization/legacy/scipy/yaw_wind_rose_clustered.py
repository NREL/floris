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

import copy

import numpy as np
import pandas as pd

from ....logging_manager import LoggerBase
from .cluster_turbines import cluster_turbines
from .yaw_wind_rose import YawOptimizationWindRose


class YawOptimizationWindRoseClustered(YawOptimizationWindRose, LoggerBase):
    """
    YawOptimizationWindRose is a subclass of
    :py:class:`~.tools.optimizationscipy.YawOptimizationWindRose` that is used
    to perform optimizations of the yaw angles of all or a subset of wind
    turbines in a Floris Farm for multiple sets of inflow conditions using the
    scipy optimization package. This class facilitates the clusterization of the
    turbines inside seperate subsets in which the turbines witin each subset
    exclusively interact with one another and have no impact on turbines
    in other clusters. This may significantly reduce the computational
    burden at no loss in performance (assuming the turbine clusters are truly
    independent).
    """

    def __init__(
        self,
        fi,
        wd,
        ws,
        ti=None,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        minimum_ws=3.0,
        maximum_ws=25.0,
        yaw_angles_baseline=None,
        x0=None,
        bnds=None,
        opt_method="SLSQP",
        opt_options=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        turbine_weights=None,
        verbose=False,
        calc_init_power=True,
        exclude_downstream_turbines=False,
        clustering_wake_slope=0.30,
    ):
        """
        Instantiate YawOptimizationWindRose object with a FlorisInterface object
        and assign parameter values.

        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
            wd (iterable) : The wind directions for which the yaw angles are
                optimized (deg).
            ws (iterable): The wind speeds for which the yaw angles are
                optimized (m/s).
            ti (iterable, optional): An optional list of turbulence intensity
                values for which the yaw angles are optimized. If not
                specified, the current TI value in the Floris object will be
                used for all optimizations. Defaults to None.
            minimum_yaw_angle (float, optional): Minimum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to 0.0.
            maximum_yaw_angle (float, optional): Maximum constraint on yaw
                angle (deg). This value will be ignored if bnds is also
                specified. Defaults to 25.0.
            minimum_ws (float, optional): Minimum wind speed at which
                optimization is performed (m/s). Assumes zero power generated
                below this value. Defaults to 3.
            maximum_ws (float, optional): Maximum wind speed at which
                optimization is performed (m/s). Assumes optimal yaw offsets
                are zero above this wind speed. Defaults to 25.
            yaw_angles_baseline (iterable, optional): The baseline yaw
                angles used to calculate the initial and baseline power
                production in the wind farm and used to normalize the cost
                function. If none are specified, this variable is set equal
                to the current yaw angles in floris. Note that this variable
                need not meet the yaw constraints specified in self.bnds,
                yet a warning is raised if it does to inform the user.
                Defaults to None.
            x0 (iterable, optional): The initial guess for the optimization
                problem. These values must meet the constraints specified
                in self.bnds. Note that, if exclude_downstream_turbines=True,
                the initial guess for any downstream turbines are ignored
                since they are not part of the optimization. Instead, the yaw
                angles for those turbines are 0.0 if that meets the lower and
                upper bound, or otherwise as close to 0.0 as feasible. If no
                values for x0 are specified, x0 is set to be equal to zeros
                wherever feasible (w.r.t. the bounds), and equal to the
                average of its lower and upper bound for all non-downstream
                turbines otherwise. Defaults to None.
            bnds (iterable, optional): Bounds for the yaw angles, as tuples of
                min, max values for each turbine (deg). One can fix the yaw
                angle of certain turbines to a predefined value by setting that
                turbine's lower bound equal to its upper bound (i.e., an
                equality constraint), as: bnds[ti] = (x, x), where x is the
                fixed yaw angle assigned to the turbine. This works for both
                zero and nonzero yaw angles. Moreover, if
                exclude_downstream_turbines=True, the yaw angles for all
                downstream turbines will be 0.0 or a feasible value closest to
                0.0. If none are specified, the bounds are set to
                (minimum_yaw_angle, maximum_yaw_angle) for each turbine. Note
                that, if bnds is not none, its values overwrite any value given
                in minimum_yaw_angle and maximum_yaw_angle. Defaults to None.
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to 'SLSQP'.
            opt_options (dictionary, optional): Optimization options used by
                scipy.optimize.minize. If none are specified, they are set to
                {'maxiter': 100, 'disp': False, 'iprint': 1, 'ftol': 1e-7,
                'eps': 0.01}. Defaults to None.
            include_unc (bool, optional): Determines whether wind direction or
                yaw uncertainty are included. If True, uncertainty in wind
                direction and/or yaw position is included when determining
                wind farm power. Uncertainty is included by computing the
                mean wind farm power for a distribution of wind direction
                and yaw position deviations from the intended wind direction
                and yaw angles. Defaults to False.
            unc_pmfs (dictionary, optional): A dictionary containing
                probability mass functions describing the distribution of
                wind direction and yaw position deviations when wind direction
                and/or yaw position uncertainty is included in the power
                calculations. Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): The wind direction
                    deviations from the intended wind direction (deg).
                -   **wd_unc_pmf** (*np.array*): The probability
                    of each wind direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): The yaw angle deviations
                    from the intended yaw angles (deg).
                -   **yaw_unc_pmf** (*np.array*): The probability
                    of each yaw angle deviation in **yaw_unc** occuring.

                If none are specified, default PMFs are calculated using
                values provided in **unc_options**. Defaults to None.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): The standard deviation of
                    the wind direction deviations from the original wind
                    direction (deg).
                -   **std_yaw** (*float*): The standard deviation of
                    the yaw angle deviations from the original yaw angles (deg).
                -   **pmf_res** (*float*): The resolution in degrees
                    of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): The cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                If none are specified, default values of
                {'std_wd': 4.95, 'std_yaw': 1.75, 'pmf_res': 1.0,
                'pdf_cutoff': 0.995} are used. Defaults to None.
            turbine_weights (iterable, optional): weighing terms that allow
                the user to emphasize power gains at particular turbines or
                completely ignore power gains from other turbines. The array
                of turbine powers from floris is multiplied with this array
                in the calculation of the objective function. If None, this
                is an array with all values 1.0 and length equal to the
                number of turbines. Defaults to None.
            calc_init_power (bool, optional): If True, calculates initial
                wind farm power for each set of wind conditions. Defaults to
                True.
            exclude_downstream_turbines (bool, optional): If True,
                automatically finds and excludes turbines that are most
                downstream from the optimization problem. This significantly
                reduces computation time at no loss in performance. The yaw
                angles of these downstream turbines are fixed to 0.0 deg if
                the yaw bounds specified in self.bnds allow that, or otherwise
                are fixed to the lower or upper yaw bound, whichever is closer
                to 0.0. Defaults to False.
            clustering_wake_slope (float, optional): linear slope of the wake
                in the simplified linear expansion wake model (dy/dx). This
                model is used to derive wake interactions between turbines and
                to identify the turbine clusters. A good value is about equal
                to the turbulence intensity in FLORIS. Though, since yaw
                optimizations may shift the wake laterally, a safer option
                is twice the turbulence intensity. The default value is 0.30
                which should be valid for yaw optimizations at wd_std = 0.0 deg
                and turbulence intensities up to 15%. Defaults to 0.30.
        """
        super().__init__(
            fi=fi,
            wd=wd,
            ws=ws,
            ti=ti,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            minimum_ws=minimum_ws,
            maximum_ws=maximum_ws,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            bnds=bnds,
            opt_method=opt_method,
            opt_options=opt_options,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            turbine_weights=turbine_weights,
            verbose=verbose,
            calc_init_power=calc_init_power,
            exclude_downstream_turbines=exclude_downstream_turbines,
        )
        self.clustering_wake_slope = clustering_wake_slope


    def _cluster_turbines(self):
        wind_directions = self.fi.floris.farm.wind_direction
        if (np.std(wind_directions) > 0.001):
            raise ValueError("Wind directions must be uniform for clustering algorithm.")
        self.clusters = cluster_turbines(
            fi=self.fi,
            wind_direction=self.fi.floris.farm.wind_direction[0],
            wake_slope=self.clustering_wake_slope
        )

    def plot_clusters(self):
        for wd in self.wd:
            cluster_turbines(
                fi=self.fi,
                wind_direction=wd,
                wake_slope=self.clustering_wake_slope,
                plot_lines=True
            )


    def optimize(self):
        """
        This method solves for the optimum turbine yaw angles for power
        production and the resulting power produced by the wind farm for a
        series of wind speed, wind direction, and optionally TI combinations.

        Returns:
            pandas.DataFrame: A pandas DataFrame with the same number of rows
            as the length of the wd and ws arrays, containing the following
            columns:

                - **ws** (*float*) - The wind speed values for which the yaw
                angles are optimized and power is computed (m/s).
                - **wd** (*float*) - The wind direction values for which the
                yaw angles are optimized and power is computed (deg).
                - **ti** (*float*) - The turbulence intensity values for which
                the yaw angles are optimized and power is computed. Only
                included if self.ti is not None.
                - **power_opt** (*float*) - The total power produced by the
                wind farm with optimal yaw offsets (W).
                - **turbine_power_opt** (*list* (*float*)) - A list
                containing the power produced by each wind turbine with optimal
                yaw offsets (W).
                - **yaw_angles** (*list* (*float*)) - A list containing
                the optimal yaw offsets for maximizing total wind farm power
                for each wind turbine (deg).
        """
        print("=====================================================")
        print("Optimizing wake redirection control...")
        print("Number of wind conditions to optimize = ", len(self.wd))
        print("Number of yaw angles to optimize = ", len(self.turbs_to_opt))
        print("=====================================================")

        df_opt = pd.DataFrame()

        for i in range(len(self.wd)):
            if self.verbose:
                if self.ti is None:
                    print(
                        "Computing wind speed, wind direction pair "
                        + str(i)
                        + " out of "
                        + str(len(self.wd))
                        + ": wind speed = "
                        + str(self.ws[i])
                        + " m/s, wind direction = "
                        + str(self.wd[i])
                        + " deg."
                    )
                else:
                    print(
                        "Computing wind speed, wind direction, turbulence "
                        + "intensity set "
                        + str(i)
                        + " out of "
                        + str(len(self.wd))
                        + ": wind speed = "
                        + str(self.ws[i])
                        + " m/s, wind direction = "
                        + str(self.wd[i])
                        + " deg, turbulence intensity = "
                        + str(self.ti[i])
                        + "."
                    )

            # Optimizing wake redirection control
            if (self.ws[i] >= self.minimum_ws) & (self.ws[i] <= self.maximum_ws):
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )

                # Set initial farm power
                self.initial_farm_power = self.initial_farm_powers[i]

                # Determine clusters and then optimize by cluster
                self._cluster_turbines()
                if self.verbose:
                    print("Clustered turbines into %d separate clusters." % len(self.clusters))

                # Save parameters to a full list
                yaw_angles_template_full = copy.copy(self.yaw_angles_template)
                yaw_angles_baseline_full = copy.copy(self.yaw_angles_baseline)
                turbine_weights_full = copy.copy(self.turbine_weights)
                bnds_full = copy.copy(self.bnds)
                # nturbs_full = copy.copy(self.nturbs)
                x0_full = copy.copy(self.x0)
                fi_full = copy.deepcopy(self.fi)

                # Overwrite parameters for each cluster and optimize
                opt_yaw_angles = np.zeros_like(x0_full)
                for ci, cl in enumerate(self.clusters):
                    if self.verbose:
                        print("=====================================================")
                        print("Optimizing %d parameters in cluster %d." % (len(cl), ci))
                        print("=====================================================")
                    self.yaw_angles_template = np.array(yaw_angles_template_full)[cl]
                    self.yaw_angles_baseline = np.array(yaw_angles_baseline_full)[cl]
                    self.turbine_weights = np.array(turbine_weights_full)[cl]
                    self.bnds = np.array(bnds_full)[cl]
                    self.x0 = np.array(x0_full)[cl]
                    self.fi = copy.deepcopy(fi_full)
                    self.fi.reinitialize_flow_field(
                        layout_array=[
                            np.array(fi_full.layout_x)[cl],
                            np.array(fi_full.layout_y)[cl]
                        ]
                    )
                    opt_yaw_angles[cl] = self._optimize()

                # Restore parameters
                self.yaw_angles_template = yaw_angles_template_full
                self.yaw_angles_baseline = yaw_angles_baseline_full
                self.turbine_weights = turbine_weights_full
                self.bnds = bnds_full
                self.x0 = x0_full
                self.fi = fi_full
                self.fi.reinitialize_flow_field(
                    layout_array=[
                        np.array(fi_full.layout_x),
                        np.array(fi_full.layout_y)
                    ]
                )

                if np.sum(np.abs(opt_yaw_angles)) == 0:
                    print(
                        "No change in controls suggested for this inflow \
                        condition..."
                    )

                # optimized power
                self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
                power_opt = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )
            elif self.ws[i] >= self.maximum_ws:
                print(
                    "No change in controls suggested for this inflow \
                        condition..."
                )
                if self.ti is None:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]], wind_speed=[self.ws[i]]
                    )
                else:
                    self.fi.reinitialize_flow_field(
                        wind_direction=[self.wd[i]],
                        wind_speed=[self.ws[i]],
                        turbulence_intensity=self.ti[i],
                    )
                opt_yaw_angles = np.array(self.yaw_angles_template, copy=True)
                self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
                power_opt = self.fi.get_turbine_power(
                    include_unc=self.include_unc,
                    unc_pmfs=self.unc_pmfs,
                    unc_options=self.unc_options,
                )
            else:
                print(
                    "No change in controls suggested for this inflow \
                        condition..."
                )
                opt_yaw_angles = np.array(self.yaw_angles_template, copy=True)
                power_opt = self.nturbs * [0.0]

            # Include turbine weighing terms
            power_opt = np.multiply(self.turbine_weights, power_opt)

            # add variables to dataframe
            if self.ti is None:
                df_opt = df_opt.append(
                    pd.DataFrame(
                        {
                            "ws": [self.ws[i]],
                            "wd": [self.wd[i]],
                            "power_opt": [np.sum(power_opt)],
                            "turbine_power_opt": [power_opt],
                            "yaw_angles": [opt_yaw_angles],
                        }
                    )
                )
            else:
                df_opt = df_opt.append(
                    pd.DataFrame(
                        {
                            "ws": [self.ws[i]],
                            "wd": [self.wd[i]],
                            "ti": [self.ti[i]],
                            "power_opt": [np.sum(power_opt)],
                            "turbine_power_opt": [power_opt],
                            "yaw_angles": [opt_yaw_angles],
                        }
                    )
                )

        df_opt.reset_index(drop=True, inplace=True)

        return df_opt
