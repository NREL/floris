from __future__ import annotations

import copy
import warnings
from pathlib import Path
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris.floris_model import FlorisModel


class ParallelFlorisModel(FlorisModel):
    """
    This class mimics the FlorisModel, but enables parallelization of the main
    computational effort.
    """

    def __init__(
        self,
        configuration: dict | str | Path,
        interface: str | None = "multiprocessing",
        max_workers: int = -1,
        n_wind_condition_splits: int = 1,
        return_turbine_powers_only: bool = False,
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
        """
        # Instantiate the underlying FlorisModel
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
            warnings.warn(
                "No parallelization interface specified. Running in serial mode."
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
        else:
            multiargs = self._preprocessing()
            if self.interface is None:
                super().run()
            elif self.interface == "multiprocessing":
                with self._PoolExecutor(self.max_workers) as p:
                    p.starmap(_parallel_run, multiargs)

    def get_turbine_powers(self):
        """
        Calculates the power at each turbine in the wind farm.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """
        if self.return_turbine_powers_only:
            return self._stored_turbine_powers
        else:
            return super().get_turbine_powers()

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

def _parallel_run(fmodel_dict, **set_kwargs) -> FlorisModel:
    """
    Run the FLORIS model in parallel.

    Args:
        fmodel: The FLORIS model to run.
        kwargs: Additional keyword arguments to pass to the model.
    """
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    fmodel.run()
    # Not sure that'll work---can't be serialized? That could be a problem.
    return fmodel
