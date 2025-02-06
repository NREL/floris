from pathlib import Path

import numpy as np
import pytest

from floris import (
    FlorisModel,
    TimeSeries,
)
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.heterogeneous_map import HeterogeneousMap


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

N = 100


def test_benchmark_set(benchmark):
    fmodel = FlorisModel(configuration=YAML_INPUT)
    wind_directions = np.linspace(0, 360, N)
    wind_speeds = np.ones(N) * 8
    turbulence_intensities = np.ones(N) * 0.06

    benchmark(
        fmodel.set,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=turbulence_intensities,
    )


def test_benchmark_run(benchmark):
    fmodel = FlorisModel(configuration=YAML_INPUT)
    wind_directions = np.linspace(0, 360, N)
    wind_speeds = np.ones(N) * 8
    turbulence_intensities = np.ones(N) * 0.06

    fmodel.set(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=turbulence_intensities,
    )

    benchmark(fmodel.run)


def test_benchmark_100_turbine_run(benchmark):
    fmodel = FlorisModel(configuration=YAML_INPUT)
    wind_directions = np.linspace(0, 360, N)
    wind_speeds = np.ones(N) * 8
    turbulence_intensities = np.ones(N) * 0.06

    fmodel.set(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=turbulence_intensities,
        layout_x=np.linspace(0, 1000, 100),
        layout_y=np.linspace(0, 1000, 100),
    )

    benchmark(fmodel.run)


def test_benchmark_het_set(benchmark):
    # Define a 2D het map and confirm the results are as expected
    # when applied to FLORIS

    # The side of the flow which is accelerated reverses for east versus west
    het_map = HeterogeneousMap(
        x=np.array([0.0, 0.0, 500.0, 500.0]),
        y=np.array([0.0, 500.0, 0.0, 500.0]),
        speed_multipliers=np.array(
            [
                [1.0, 2.0, 1.0, 2.0],  # Top accelerated
                [2.0, 1.0, 2.0, 1.0],  # Bottom accelerated
            ]
        ),
        wind_directions=np.array([270.0, 90.0]),
        wind_speeds=np.array([8.0, 8.0]),
    )

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    time_series = TimeSeries(
        wind_directions=np.linspace(0, 360, N),
        wind_speeds=8.0,
        turbulence_intensities=0.06,
        heterogeneous_map=het_map,
    )

    # Set the model to a turbines perpinducluar to
    # east/west flow with 0 turbine closer to bottom and
    # turbine 1 closer to top
    benchmark(
        fmodel.set,
        wind_data=time_series,
        layout_x=[250.0, 250.0],
        layout_y=[100.0, 400.0],
    )


def test_benchmark_het_run(benchmark):
    # Define a 2D het map and confirm the results are as expected
    # when applied to FLORIS

    # The side of the flow which is accelerated reverses for east versus west
    het_map = HeterogeneousMap(
        x=np.array([0.0, 0.0, 500.0, 500.0]),
        y=np.array([0.0, 500.0, 0.0, 500.0]),
        speed_multipliers=np.array(
            [
                [1.0, 2.0, 1.0, 2.0],  # Top accelerated
                [2.0, 1.0, 2.0, 1.0],  # Bottom accelerated
            ]
        ),
        wind_directions=np.array([270.0, 90.0]),
        wind_speeds=np.array([8.0, 8.0]),
    )

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    time_series = TimeSeries(
        wind_directions=np.linspace(0, 360, N),
        wind_speeds=8.0,
        turbulence_intensities=0.06,
        heterogeneous_map=het_map,
    )

    # Set the model to a turbines perpinducluar to
    # east/west flow with 0 turbine closer to bottom and
    # turbine 1 closer to top
    fmodel.set(
        wind_data=time_series,
        layout_x=[250.0, 250.0],
        layout_y=[100.0, 400.0],
    )

    benchmark(fmodel.run)
