# Copyright 2022 Shell
import copy
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris.logging_manager import LoggerBase
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.tools.uncertainty_interface import FlorisInterface, UncertaintyInterface


def _load_local_floris_object(
    fi_dict,
    het_map=None,
    unc_pmfs=None,
    fix_yaw_in_relative_frame=False
):
    # Load local FLORIS object
    if unc_pmfs is None:
        fi = FlorisInterface(fi_dict, het_map=het_map)
    else:
        fi = UncertaintyInterface(
            fi_dict,
            het_map=het_map,
            unc_pmfs=unc_pmfs,
            fix_yaw_in_relative_frame=fix_yaw_in_relative_frame,
        )
    return fi


def _get_turbine_powers_serial(fi_information, yaw_angles=None):
    fi = _load_local_floris_object(*fi_information)
    fi.calculate_wake(yaw_angles=yaw_angles)
    return (fi.get_turbine_powers(), fi.floris.flow_field)


def _optimize_yaw_angles_serial(
    fi_information,
    minimum_yaw_angle,
    maximum_yaw_angle,
    yaw_angles_baseline,
    x0,
    Ny_passes,
    turbine_weights,
    exclude_downstream_turbines,
    exploit_layout_symmetry,
    verify_convergence,
    print_progress,
):
    fi_opt = _load_local_floris_object(*fi_information)
    yaw_opt = YawOptimizationSR(
        fi=fi_opt,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
        yaw_angles_baseline=yaw_angles_baseline,
        x0=x0,
        Ny_passes=Ny_passes,
        turbine_weights=turbine_weights,
        exclude_downstream_turbines=exclude_downstream_turbines,
        exploit_layout_symmetry=exploit_layout_symmetry,
        verify_convergence=verify_convergence,
    )

    # Perform optimization but silence print statements to avoid cluttering
    df_opt = yaw_opt.optimize(print_progress=print_progress)
    return df_opt


class ParallelComputingInterface(LoggerBase):
    def __init__(
        self,
        fi,
        max_workers,
        n_wind_direction_splits,
        n_wind_speed_splits=1,
        use_mpi4py=False,
        propagate_flowfield_from_workers=False,
        print_timings=False
    ):
        """A wrapper around the nominal floris_interface class that adds
        parallel computing to common FlorisInterface properties.

        Args:
        fi (FlorisInterface or UncertaintyInterface object): Interactive FLORIS object used to
            perform the wake and turbine calculations. Can either be a regular FlorisInterface
            object or can be an UncertaintyInterface object.
        """

        # Load the correct library
        if use_mpi4py:
            import mpi4py.futures as mp
            self._PoolExecutor = mp.MPIPoolExecutor
        else:
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers is None:
                max_workers = mp.cpu_count()

        # Initialize floris object and copy common properties
        self.fi = fi.copy()
        self.floris = self.fi.floris  # Static copy as a placeholder

        # Save to self
        self._n_wind_direction_splits = n_wind_direction_splits  # Save initial user input
        self._n_wind_speed_splits = n_wind_speed_splits  # Save initial user input
        self._max_workers = max_workers  # Save initial user input

        self.n_wind_direction_splits = int(
            np.min([n_wind_direction_splits, self.fi.floris.flow_field.n_wind_directions])
        )
        self.n_wind_speed_splits = int(
            np.min([n_wind_speed_splits, self.fi.floris.flow_field.n_wind_speeds])
        )
        self.max_workers = int(
            np.min([max_workers, self.n_wind_direction_splits * self.n_wind_speed_splits])
        )
        self.propagate_flowfield_from_workers = propagate_flowfield_from_workers
        self.use_mpi4py = use_mpi4py
        self.print_timings = print_timings

    def copy(self):
        # Make an independent copy
        self_copy = copy.deepcopy(self)
        self_copy.fi = self.fi.copy()
        return self_copy

    def reinitialize(
        self,
        wind_speeds=None,
        wind_directions=None,
        wind_shear=None,
        wind_veer=None,
        reference_wind_height=None,
        turbulence_intensity=None,
        air_density=None,
        layout=None,
        layout_x=None,
        layout_y=None,
        turbine_type=None,
        solver_settings=None,
    ):
        """Pass to the FlorisInterface reinitialize function. To allow users
        to directly replace a FlorisInterface object with this
        UncertaintyInterface object, this function is required."""

        if layout is not None:
            msg = "Use the `layout_x` and `layout_y` parameters in place of `layout` "
            msg += "because the `layout` parameter will be deprecated in 3.3."
            self.logger.warning(msg)
            layout_x = layout[0]
            layout_y = layout[1]

        # Just passes arguments to the floris object
        fi = self.fi.copy()
        fi.reinitialize(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            reference_wind_height=reference_wind_height,
            turbulence_intensity=turbulence_intensity,
            air_density=air_density,
            layout_x=layout_x,
            layout_y=layout_y,
            turbine_type=turbine_type,
            solver_settings=solver_settings,
        )

        # Reinitialize settings
        self.__init__(
            fi=fi,
            max_workers=self._max_workers,
            n_wind_direction_splits=self._n_wind_direction_splits,
            n_wind_speed_splits=self._n_wind_speed_splits,
            use_mpi4py=self.use_mpi4py,
            propagate_flowfield_from_workers=self.propagate_flowfield_from_workers,
            print_timings=self.print_timings,
        )

    def _preprocessing(self, yaw_angles=None):
        # Format yaw angles
        if yaw_angles is None:
            yaw_angles = np.zeros((
                self.fi.floris.flow_field.n_wind_directions,
                self.fi.floris.flow_field.n_wind_speeds,
                self.fi.floris.farm.n_turbines
            ))

        # Prepare settings
        n_wind_direction_splits = self.n_wind_direction_splits
        n_wind_direction_splits = np.min(
            [n_wind_direction_splits, self.fi.floris.flow_field.n_wind_directions]
        )
        n_wind_speed_splits = self.n_wind_speed_splits
        n_wind_speed_splits = np.min([n_wind_speed_splits, self.fi.floris.flow_field.n_wind_speeds])

        # Prepare the input arguments for parallel execution
        fi_dict = self.fi.floris.as_dict()
        wind_direction_id_splits = np.array_split(
            np.arange(self.fi.floris.flow_field.n_wind_directions),
            n_wind_direction_splits
        )
        wind_speed_id_splits = np.array_split(
            np.arange(self.fi.floris.flow_field.n_wind_speeds),
            n_wind_speed_splits
        )
        multiargs = []
        for wd_id_split in wind_direction_id_splits:
            for ws_id_split in wind_speed_id_splits:
                fi_dict_split = copy.deepcopy(fi_dict)
                wind_directions = self.fi.floris.flow_field.wind_directions[wd_id_split]
                wind_speeds = self.fi.floris.flow_field.wind_speeds[ws_id_split]
                yaw_angles_subset = yaw_angles[wd_id_split[0]:wd_id_split[-1]+1, ws_id_split, :]
                fi_dict_split["flow_field"]["wind_directions"] = wind_directions
                fi_dict_split["flow_field"]["wind_speeds"] = wind_speeds

                # Prepare lightweight data to pass along
                if isinstance(self.fi, FlorisInterface):
                    fi_information = (fi_dict_split, self.fi.het_map, None, None)
                else:
                    fi_information = (
                        fi_dict_split,
                        self.fi.fi.het_map,
                        self.fi.unc_pmfs,
                        self.fi.fix_yaw_in_relative_frame
                    )
                multiargs.append((fi_information, yaw_angles_subset))

        return multiargs

    # Function to merge subsets in dictionaries
    def _merge_subsets(self, field, subset):
        return np.concatenate(  # Merges wind speeds
            [
                np.concatenate(  # Merges wind directions
                    [
                        eval("f.{:s}".format(field))
                        for f in subset[
                            wii
                            * self.n_wind_direction_splits:(wii+1)
                            * self.n_wind_direction_splits
                        ]
                    ],
                    axis=0
                )
                for wii in range(self.n_wind_speed_splits)
            ],
            axis=1
        )

    def _postprocessing(self, output):
        # Split results
        power_subsets = [p[0] for p in output]
        flowfield_subsets = [p[1] for p in output]

        # Retrieve and merge turbine power productions
        turbine_powers = np.concatenate(
            [
                np.concatenate(
                    power_subsets[self.n_wind_speed_splits*(ii):self.n_wind_speed_splits*(ii+1)],
                    axis=1
                )
                for ii in range(self.n_wind_direction_splits)
            ],
            axis=0
        )

        # Optionally, also merge flow field dictionaries from individual floris solutions
        if self.propagate_flowfield_from_workers:
            self.floris = self.fi.floris  # Refresh static copy of underlying floris class
            # self.floris.flow_field.u_initial = self._merge_subsets("u_initial", flowfield_subsets)
            # self.floris.flow_field.v_initial = self._merge_subsets("v_initial", flowfield_subsets)
            # self.floris.flow_field.w_initial = self._merge_subsets("w_initial", flowfield_subsets)
            self.floris.flow_field.u = self._merge_subsets("u", flowfield_subsets)
            self.floris.flow_field.v = self._merge_subsets("v", flowfield_subsets)
            self.floris.flow_field.w = self._merge_subsets("w", flowfield_subsets)
            self.floris.flow_field.turbulence_intensity_field = self._merge_subsets(
                "turbulence_intensity_field",
                flowfield_subsets
            )

        return turbine_powers

    def calculate_wake(self):
        # raise UserWarning("'calculate_wake' not supported. Please use
        # 'get_turbine_powers' or 'get_farm_power' directly.")
        return None  # Do nothing

    def get_turbine_powers(self, yaw_angles=None):
        # Retrieve multiargs: preprocessing
        t0 = timerpc()
        multiargs = self._preprocessing(yaw_angles)
        t_preparation = timerpc() - t0

        # Perform parallel calculation
        t1 = timerpc()
        with self._PoolExecutor(self.max_workers) as p:
            out = p.starmap(_get_turbine_powers_serial, multiargs)
        t_execution = timerpc() - t1

        # Postprocessing: merge power production (and opt. flow field) from individual runs
        t2 = timerpc()
        turbine_powers = self._postprocessing(out)
        t_postprocessing = timerpc() - t2
        t_total = timerpc() - t0

        if self.print_timings:
            print("===============================================================================")
            print(
                "Total time spent for parallel calculation "
                f"({self.max_workers} workers): {t_total:.3f} s"
            )
            print(f"  Time spent in parallel preprocessing: {t_preparation:.3f} s")
            print(f"  Time spent in parallel loop execution: {t_execution:.3f} s.")
            print(f"  Time spent in parallel postprocessing: {t_postprocessing:.3f} s")

        return turbine_powers

    def get_farm_power(self, yaw_angles=None, turbine_weights=None):
        if turbine_weights is None:
            # Default to equal weighing of all turbines when turbine_weights is None
            turbine_weights = np.ones(
                (
                    self.fi.floris.flow_field.n_wind_directions,
                    self.fi.floris.flow_field.n_wind_speeds,
                    self.fi.floris.farm.n_turbines
                )
            )
        elif len(np.shape(turbine_weights)) == 1:
            # Deal with situation when 1D array is provided
            turbine_weights = np.tile(
                turbine_weights,
                (
                    self.fi.floris.flow_field.n_wind_directions,
                    self.fi.floris.flow_field.n_wind_speeds,
                    1
                )
            )

        # Calculate all turbine powers and apply weights
        turbine_powers = self.get_turbine_powers(yaw_angles=yaw_angles)
        turbine_powers = np.multiply(turbine_weights, turbine_powers)

        return np.sum(turbine_powers, axis=2)

    def get_farm_AEP(
        self,
        freq,
        cut_in_wind_speed=0.001,
        cut_out_wind_speed=None,
        yaw_angles=None,
        turbine_weights=None,
        no_wake=False,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, frequency of occurrence, and yaw offset.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_wind_directions,
                n_wind_speeds) with the frequencies of each wind direction and
                wind speed combination. These frequencies should typically sum
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP.
            cut_in_wind_speed (float, optional): Wind speed in m/s below which
                any calculations are ignored and the wind farm is known to
                produce 0.0 W of power. Note that to prevent problems with the
                wake models at negative / zero wind speeds, this variable must
                always have a positive value. Defaults to 0.001 [m/s].
            cut_out_wind_speed (float, optional): Wind speed above which the
                wind farm is known to produce 0.0 W of power. If None is
                specified, will assume that the wind farm does not cut out
                at high wind speeds. Defaults to None.
            yaw_angles (NDArrayFloat | list[float] | None, optional):
                The relative turbine yaw angles in degrees. If None is
                specified, will assume that the turbine yaw angles are all
                zero degrees for all conditions. Defaults to None.
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_wind_directions, n_wind_speeds,
                n_turbines). Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the wake to
                the flow field. This can be useful when quantifying the loss
                in AEP due to wakes. Defaults to *False*.

        Returns:
            float:
                The Annual Energy Production (AEP) for the wind farm in
                watt-hours.
        """

        # If no_wake==True, ignore parallelization because it's fast enough
        if no_wake:
            return self.fi.get_farm_AEP(
                freq=freq,
                cut_in_wind_speed=cut_in_wind_speed,
                cut_out_wind_speed=cut_out_wind_speed,
                yaw_angles=yaw_angles,
                turbine_weights=turbine_weights,
                no_wake=no_wake
            )

        # Verify dimensions of the variable "freq"
        if not (
            (np.shape(freq)[0] == self.fi.floris.flow_field.n_wind_directions)
            & (np.shape(freq)[1] == self.fi.floris.flow_field.n_wind_speeds)
            & (len(np.shape(freq)) == 2)
        ):
            raise UserWarning(
                "'freq' should be a two-dimensional array with dimensions "
                + "(n_wind_directions, n_wind_speeds)."
            )

        # Check if frequency vector sums to 1.0. If not, raise a warning
        if np.abs(np.sum(freq) - 1.0) > 0.001:
            self.logger.warning(
                "WARNING: The frequency array provided to get_farm_AEP() does not sum to 1.0. "
            )

        # Copy the full wind speed array from the floris object and initialize
        # the the farm_power variable as an empty array.
        wind_speeds = np.array(self.fi.floris.flow_field.wind_speeds, copy=True)
        farm_power = np.zeros((self.fi.floris.flow_field.n_wind_directions, len(wind_speeds)))

        # Determine which wind speeds we must evaluate in floris
        conditions_to_evaluate = wind_speeds >= cut_in_wind_speed
        if cut_out_wind_speed is not None:
            conditions_to_evaluate = conditions_to_evaluate & (wind_speeds < cut_out_wind_speed)

        # Evaluate the conditions in floris
        if np.any(conditions_to_evaluate):
            wind_speeds_subset = wind_speeds[conditions_to_evaluate]
            yaw_angles_subset = None
            if yaw_angles is not None:
                yaw_angles_subset = yaw_angles[:, conditions_to_evaluate]
            self.fi.reinitialize(wind_speeds=wind_speeds_subset)
            farm_power[:, conditions_to_evaluate] = (
                self.get_farm_power(yaw_angles=yaw_angles_subset, turbine_weights=turbine_weights)
            )

        # Finally, calculate AEP in GWh
        aep = np.sum(np.multiply(freq, farm_power) * 365 * 24)

        # Reset the FLORIS object to the full wind speed array
        self.fi.reinitialize(wind_speeds=wind_speeds)

        return aep

    def optimize_yaw_angles(
        self,
        minimum_yaw_angle=-25.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5,4],
        turbine_weights=None,
        exclude_downstream_turbines=True,
        exploit_layout_symmetry=True,
        verify_convergence=False,
        print_worker_progress=False,  # Recommended disabled to avoid clutter. Useful for debugging
    ):

        # Prepare the inputs to each core for multiprocessing module
        t0 = timerpc()
        multiargs = self._preprocessing()
        for ii in range(len(multiargs)):
            multiargs[ii] = (
                multiargs[ii][0],
                minimum_yaw_angle,
                maximum_yaw_angle,
                yaw_angles_baseline,
                x0,
                Ny_passes,
                turbine_weights,
                exclude_downstream_turbines,
                exploit_layout_symmetry,
                verify_convergence,
                print_worker_progress,
            )
        t1 = timerpc()

        # Optimize yaw angles using parallel processing
        print("Optimizing yaw angles with {:d} workers.".format(self.max_workers))
        with self._PoolExecutor(self.max_workers) as p:
            df_opt_splits = p.starmap(_optimize_yaw_angles_serial, multiargs)
        t2 = timerpc()

        # Combine all solutions from multiprocessing into single dataframe
        df_opt = pd.concat(df_opt_splits, axis=0).reset_index(drop=True).sort_values(
            by=["wind_direction", "wind_speed", "turbulence_intensity"]
        )
        t3 = timerpc()

        if self.print_timings:
            print("===============================================================================")
            print(
                "Total time spent for parallel calculation "
                f"({self.max_workers} workers): {t3 - t0:.3f} s"
            )
            print("  Time spent in parallel preprocessing: {:.3f} s".format(t1 - t0))
            print("  Time spent in parallel loop execution: {:.3f} s.".format(t2 - t1))
            print("  Time spent in parallel postprocessing: {:.3f} s".format(t3 - t2))

        return df_opt

    @property
    def layout_x(self):
        return self.fi.layout_x

    @property
    def layout_y(self):
        return self.fi.layout_y

    # @property
    # def floris(self):
    #     return self.fi.floris
