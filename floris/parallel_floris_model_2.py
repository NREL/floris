from __future__ import annotations

import copy
import warnings
from pathlib import Path
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris.core import State
from floris.floris_model import FlorisModel
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


class ParallelFlorisModel(FlorisModel):
    """
    This class mimics the FlorisModel, but enables parallelization of the main
    computational effort.
    """

    def __init__(
        self,
        configuration: dict | str | Path | FlorisModel,
        interface: str | None = "multiprocessing",
        max_workers: int = -1,
        n_wind_condition_splits: int = 1,
        return_turbine_powers_only: bool = False,
        print_timings: bool = False
    ):
        """
        Initialize the ParallelFlorisModel object.

        Args:
            configuration: The Floris configuration dictionary or YAML file.
            The configuration should have the following inputs specified.
                - **flow_field**: See `floris.simulation.flow_field.FlowField` for more details.
                - **farm**: See `floris.simulation.farm.Farm` for more details.
                - **turbine**: See `floris.simulation.turbine.Turbine` for more details.
                - **wake**: See `floris.simulation.wake.WakeManager` for more details.
                - **logging**: See `floris.simulation.core.Core` for more details.
            interface: The parallelization interface to use. Options are "multiprocessing",
               with possible future support for "mpi4py" and "concurrent"
            max_workers: The maximum number of workers to use. Defaults to -1, which then
               takes the number of CPUs available.
            n_wind_condition_splits: The number of wind conditions to split the simulation over.
               Defaults to the same as max_workers.
            return_turbine_powers_only: Whether to return only the turbine powers.
            print_timings (bool): Print the computation time to the console. Defaults to False.
        """
        # Instantiate the underlying FlorisModel
        if isinstance(configuration, FlorisModel):
            self.logger.warning(
                "Received an instantiated FlorisModel, when expected a dictionary or path"
                " to a FLORIS input file. Converting to dictionary to instantiate "
                " the ParallelFlorisModel."
            )
            configuration = configuration.core.as_dict()
        super().__init__(configuration)

        # Save parallelization parameters
        if interface == "multiprocessing":
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers == -1:
                max_workers = mp.cpu_count()
            # TODO: test spinning up the worker pool at this point
        elif interface in ["mpi4py", "concurrent"]:
            raise NotImplementedError(
                f"Parallelization interface {interface} not yet supported."
            )
        elif interface is None:
            self.logger.warning(
                "No parallelization interface specified. Running in serial mode."
            )
            if return_turbine_powers_only:
                self.logger.warn(
                    "return_turbine_powers_only is not supported in serial mode."
                )
        else:
            raise ValueError(
                f"Invalid parallelization interface {interface}. "
                "Options are 'multiprocessing', 'mpi4py' or 'concurrent'."
            )

        self.interface = interface
        self.max_workers = max_workers
        if n_wind_condition_splits == -1:
            self.n_wind_condition_splits = max_workers
        else:
            self.n_wind_condition_splits = n_wind_condition_splits
        self.return_turbine_powers_only = return_turbine_powers_only
        self.print_timings = print_timings

    def run(self) -> None:
        """
        Run the FLORIS model in parallel.
        """

        if self.return_turbine_powers_only:
            # TODO: code here that does not return flow fields
            # Somehow, overload methods on FlorisModel that need flow field
            # data.

            # This version will call super().get_turbine_powers() on each of
            # the splits, and return them somehow.
            self._stored_turbine_powers = None # Temporary
        if self.interface is None:
            t0 = timerpc()
            super().run()
            t1 = timerpc()
        elif self.interface == "multiprocessing":
            t0 = timerpc()
            parallel_run_inputs = self._preprocessing()
            t1 = timerpc()
            if self.return_turbine_powers_only:
                with self._PoolExecutor(self.max_workers) as p:
                    self._turbine_powers_split = p.starmap(
                        _parallel_run_powers_only,
                        parallel_run_inputs
                    )
            else:
                with self._PoolExecutor(self.max_workers) as p:
                    self._fmodels_split = p.starmap(_parallel_run, parallel_run_inputs)
            t2 = timerpc()
            self._postprocessing()
            self.core.farm.finalize(self.core.grid.unsorted_indices)
            self.core.state = State.USED
            t3 = timerpc()
        if self.print_timings:
            print("===============================================================================")
            if self.interface is None:
                print(f"Total time spent for serial calculation (interface=None): {t1 - t0:.3f} s")
            else:
                print(
                    "Total time spent for parallel calculation "
                    f"({self.max_workers} workers): {t3-t0:.3f} s"
                )
                print(f"  Time spent in parallel preprocessing: {t1-t0:.3f} s")
                print(f"  Time spent in parallel loop execution: {t2-t1:.3f} s.")
                print(f"  Time spent in parallel postprocessing: {t3-t2:.3f} s")

    def _preprocessing(self):
        # Split over the wind conditions
        n_wind_condition_splits = self.n_wind_condition_splits
        n_wind_condition_splits = np.min(
            [n_wind_condition_splits, self.core.flow_field.n_findex]
        )

        # Prepare the input arguments for parallel execution
        fmodel_dict = self.core.as_dict()
        wind_condition_id_splits = np.array_split(
            np.arange(self.core.flow_field.n_findex),
            n_wind_condition_splits,
        )
        multiargs = []
        for wc_id_split in wind_condition_id_splits:
            # for ws_id_split in wind_speed_id_splits:
            fmodel_dict_split = copy.deepcopy(fmodel_dict)
            wind_directions = self.core.flow_field.wind_directions[wc_id_split]
            wind_speeds = self.core.flow_field.wind_speeds[wc_id_split]
            turbulence_intensities = self.core.flow_field.turbulence_intensities[wc_id_split]

            # Extract and format all control setpoints as a dict that can be unpacked later
            control_setpoints_subset = {
                "yaw_angles": self.core.farm.yaw_angles[wc_id_split, :],
                "power_setpoints": self.core.farm.power_setpoints[wc_id_split, :],
                "awc_modes": self.core.farm.awc_modes[wc_id_split, :],
                "awc_amplitudes": self.core.farm.awc_amplitudes[wc_id_split, :],
                "awc_frequencies": self.core.farm.awc_frequencies[wc_id_split, :],
            }

            # Do I also need to hangle splitting the wind data object?
            # Perhaps not, not used in run() I think

            fmodel_dict_split["flow_field"]["wind_directions"] = wind_directions
            fmodel_dict_split["flow_field"]["wind_speeds"] = wind_speeds
            fmodel_dict_split["flow_field"]["turbulence_intensities"] = turbulence_intensities

            # Prepare lightweight data to pass along
            multiargs.append((fmodel_dict_split, control_setpoints_subset))

        return multiargs

    def _postprocessing(self):
        # Append the remaining flow_fields
        # Could consider adding a merge method to the FlowField class
        # to make this easier

        if self.return_turbine_powers_only:
            self._stored_turbine_powers = np.vstack(self._turbine_powers_split)
        else:
            # Ensure fields to set have correct dimensions
            self.core.flow_field.u = self._fmodels_split[0].core.flow_field.u
            self.core.flow_field.v = self._fmodels_split[0].core.flow_field.v
            self.core.flow_field.w = self._fmodels_split[0].core.flow_field.w
            self.core.flow_field.turbulence_intensity_field = \
                self._fmodels_split[0].core.flow_field.turbulence_intensity_field

            for fm in self._fmodels_split[1:]:
                self.core.flow_field.u = np.append(
                    self.core.flow_field.u,
                    fm.core.flow_field.u,
                    axis=0
                )
                self.core.flow_field.v = np.append(
                    self.core.flow_field.v,
                    fm.core.flow_field.v,
                    axis=0
                )
                self.core.flow_field.w = np.append(
                    self.core.flow_field.w,
                    fm.core.flow_field.w,
                    axis=0
                )
                self.core.flow_field.turbulence_intensity_field = np.append(
                    self.core.flow_field.turbulence_intensity_field,
                    fm.core.flow_field.turbulence_intensity_field,
                    axis=0
                )

    def _get_turbine_powers(self):
        """
        Calculates the power at each turbine in the wind farm.
        This override will only be necessary if we want to be able to
        use the return_turbine_powers_only option. Need to check if that
        makes a significant speed difference.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """
        if self.core.state is not State.USED:
            self.logger.warning(
                f"Please call `{self.__class__.__name__}.run` before computing"
                " turbine powers. In future versions, an explicit run() call will"
                "be required."
            )
            self.run()
        if self.return_turbine_powers_only:
            return self._stored_turbine_powers
        else:
            return super()._get_turbine_powers()

    @property
    def fmodel(self):
        """
        Raise deprecation warning.
        """
        self.logger.warning(
            "ParallelFlorisModel no longer contains `fmodel` as an attribute "
            "and now directly inherits from FlorisModel. Please use the "
            "attributes and methods of FlorisModel directly."
        )


def _parallel_run(fmodel_dict, set_kwargs) -> FlorisModel:
    """
    Run the FLORIS model in parallel.

    Args:
        fmodel: The FLORIS model to run.
        set_kwargs: Additional keyword arguments to pass to fmodel.set().
    """
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    fmodel.run()
    return fmodel

def _parallel_run_powers_only(fmodel_dict, set_kwargs) -> np.ndarray:
    """
    Run the FLORIS model in parallel, returning only the turbine powers.

    Args:
        fmodel: The FLORIS model to run.
        set_kwargs: Additional keyword arguments to pass to fmodel.set().
    """
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    fmodel.run()
    return fmodel.get_turbine_powers()

def _optimize_yaw_angles_serial(
    fmodel_information,
    minimum_yaw_angle,
    maximum_yaw_angle,
    yaw_angles_baseline,
    x0,
    Ny_passes,
    turbine_weights,
    exclude_downstream_turbines,
    verify_convergence,
    print_progress,
):
    fmodel_opt = FlorisModel(fmodel_information)
    yaw_opt = YawOptimizationSR(
        fmodel=fmodel_opt,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
        yaw_angles_baseline=yaw_angles_baseline,
        x0=x0,
        Ny_passes=Ny_passes,
        turbine_weights=turbine_weights,
        exclude_downstream_turbines=exclude_downstream_turbines,
        verify_convergence=verify_convergence,
    )

    # Perform optimization but silence print statements to avoid cluttering
    df_opt = yaw_opt.optimize(print_progress=print_progress)
    return df_opt