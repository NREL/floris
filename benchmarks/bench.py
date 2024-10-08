
from pathlib import Path

import numpy as np
import pytest

from floris import (
    FlorisModel,
)
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT


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
