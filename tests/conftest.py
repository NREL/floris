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


import numpy as np
import pytest
from floris.simulation import Floris
from floris.simulation import FlowField
from floris.simulation import TurbineGrid, FlowFieldGrid
from floris.utilities import Vec3


def turbines_to_array(turbine_list: list):
    return [[t.average_velocity, t.Ct, t.power, t.axial_induction] for t in turbine_list]


def assert_results_arrays(test: np.array, baseline: np.array):
    if np.shape(test) != np.shape(baseline):
        raise ValueError("test and baseline results have mismatched shapes.")

    for test_dim0, baseline_dim0 in zip(test, baseline):
        for test_dim1, baseline_dim1 in zip(test_dim0, baseline_dim0):
            assert np.allclose(test_dim1, baseline_dim1)


def assert_results(test: list, baseline: list):
    if len(test) != len(baseline):
        raise ValueError("assert_results: test and baseline results have mismatched lengths.")

    for i in range(len(test)):
        for j, (t, b) in enumerate(zip(test[i], baseline[i])):
            # print(j, t, b)
            assert t == pytest.approx(b)


def print_test_values(average_velocities: list, thrusts: list, powers: list, axial_inductions: list):
    n_wd, n_ws, n_turb = np.shape(average_velocities)
    i=0
    for j in range(n_ws):
        print("[")
        for k in range(n_turb):
            print(
                "    [{:.7f}, {:.7f}, {:.7f}, {:.7f}],".format(
                    average_velocities[i,j,k], thrusts[i,j,k], powers[i,j,k], axial_inductions[i,j,k]
                )
            )
        print("],")


WIND_DIRECTIONS = [
    270.0,
    360.0,
    285.0,
    315.0,
]
N_WIND_DIRECTIONS = len(WIND_DIRECTIONS)
WIND_SPEEDS = [
    8.0,
    9.0,
    10.0,
    11.0,
]
N_WIND_SPEEDS = len(WIND_SPEEDS)
X_COORDS = [
    0.0,
    5 * 126.0,
    10 * 126.0
]
Y_COORDS = [
    0.0,
    0.0,
    0.0
]
Z_COORDS = [
    90.0,
    90.0,
    90.0
]
N_TURBINES = len(X_COORDS)
ROTOR_DIAMETER = 126.0
TURBINE_GRID_RESOLUTION = 2


## Unit test fixtures

@pytest.fixture
def vec3_fixture():
    return Vec3([4, 4, 0])

@pytest.fixture
def flow_field_fixture(sample_inputs_fixture):
    flow_field_dict = sample_inputs_fixture.flow_field
    return FlowField.from_dict(flow_field_dict)

@pytest.fixture
def turbine_grid_fixture(sample_inputs_fixture) -> TurbineGrid:
    turbine_coordinates = [Vec3(c) for c in list(zip(X_COORDS, Y_COORDS, Z_COORDS))]

    # TODO: The TurbineGrid requires that the rotor diameters be 1d but the Farm constructs them as 3d
    #   Can we make this consistent?

    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_TURBINES) )
    return TurbineGrid(
        turbine_coordinates=turbine_coordinates,
        reference_turbine_diameter=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        wind_speeds=np.array(WIND_SPEEDS),
        grid_resolution=TURBINE_GRID_RESOLUTION
    )

@pytest.fixture
def flow_field_grid_fixture(sample_inputs_fixture) -> FlowFieldGrid:
    turbine_coordinates = [Vec3(c) for c in list(zip(X_COORDS, Y_COORDS, Z_COORDS))]
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_WIND_DIRECTIONS, N_WIND_SPEEDS, N_TURBINES) )
    return FlowFieldGrid(
        turbine_coordinates=turbine_coordinates,
        reference_turbine_diameter=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        wind_speeds=np.array(WIND_SPEEDS),
        grid_resolution=[3,2,2]
    )

@pytest.fixture
def floris_fixture():
    sample_inputs = SampleInputs()
    return Floris(sample_inputs.floris)

@pytest.fixture
def sample_inputs_fixture():
    return SampleInputs()


class SampleInputs:
    """
    SampleInputs class
    """

    def __init__(self):
        self.turbine = {
            "turbine_type": "nrel_5mw",
            "rotor_diameter": 126.0,
            "hub_height": 90.0,
            "pP": 1.88,
            "pT": 1.88,
            "generator_efficiency": 1.0,
            "power_thrust_table": {
                "power": [
                    0.000000,
                    0.000000,
                    0.178085,
                    0.289075,
                    0.349022,
                    0.384728,
                    0.406059,
                    0.420228,
                    0.428823,
                    0.433873,
                    0.436223,
                    0.436845,
                    0.436575,
                    0.436511,
                    0.436561,
                    0.436517,
                    0.435903,
                    0.434673,
                    0.433230,
                    0.430466,
                    0.378869,
                    0.335199,
                    0.297991,
                    0.266092,
                    0.238588,
                    0.214748,
                    0.193981,
                    0.175808,
                    0.159835,
                    0.145741,
                    0.133256,
                    0.122157,
                    0.112257,
                    0.103399,
                    0.095449,
                    0.088294,
                    0.081836,
                    0.075993,
                    0.070692,
                    0.065875,
                    0.061484,
                    0.057476,
                    0.053809,
                    0.050447,
                    0.047358,
                    0.044518,
                    0.041900,
                    0.039483,
                ],
                "thrust": [
                    0.0,
                    0.0,
                    0.99,
                    0.99,
                    0.97373036,
                    0.92826162,
                    0.89210543,
                    0.86100905,
                    0.835423,
                    0.81237673,
                    0.79225789,
                    0.77584769,
                    0.7629228,
                    0.76156073,
                    0.76261984,
                    0.76169723,
                    0.75232027,
                    0.74026851,
                    0.72987175,
                    0.70701647,
                    0.54054532,
                    0.45509459,
                    0.39343381,
                    0.34250785,
                    0.30487242,
                    0.27164979,
                    0.24361964,
                    0.21973831,
                    0.19918151,
                    0.18131868,
                    0.16537679,
                    0.15103727,
                    0.13998636,
                    0.1289037,
                    0.11970413,
                    0.11087113,
                    0.10339901,
                    0.09617888,
                    0.09009926,
                    0.08395078,
                    0.0791188,
                    0.07448356,
                    0.07050731,
                    0.06684119,
                    0.06345518,
                    0.06032267,
                    0.05741999,
                    0.05472609,
                ],
                "wind_speed": [
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    9.0,
                    9.5,
                    10.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    15.5,
                    16.0,
                    16.5,
                    17.0,
                    17.5,
                    18.0,
                    18.5,
                    19.0,
                    19.5,
                    20.0,
                    20.5,
                    21.0,
                    21.5,
                    22.0,
                    22.5,
                    23.0,
                    23.5,
                    24.0,
                    24.5,
                    25.0,
                    25.5,
                ],
            },
            "TSR": 8.0
        }

        self.farm = {
            "layout_x": X_COORDS,
            "layout_y": Y_COORDS,
            "turbine_type": [self.turbine]
        }

        self.flow_field = {
            "wind_speeds": WIND_SPEEDS,
            "wind_directions": WIND_DIRECTIONS,
            "turbulence_intensity": 0.1,
            "wind_shear": 0.12,
            "wind_veer": 0.0,
            "air_density": 1.225,
            "reference_wind_height": self.turbine["hub_height"],
        }

        self.wake = {
            "model_strings": {
                "velocity_model": "jensen",
                "deflection_model": "jimenez",
                "combination_model": "sosfs",
                "turbulence_model": "crespo_hernandez",
            },
            "wake_deflection_parameters": {
                "gauss": {
                    "ad": 0.0,
                    "alpha": 0.58,
                    "bd": 0.0,
                    "beta": 0.077,
                    "dm": 1.0,
                    "ka": 0.38,
                    "kb": 0.004
                },
                "jimenez": {
                    "ad": 0.0,
                    "bd": 0.0,
                    "kd": 0.05,
                },
            },
            "wake_velocity_parameters": {
                "gauss": {
                    "alpha": 0.58,
                    "beta": 0.077,
                    "ka": 0.38,
                    "kb": 0.004
                },
                "jensen": {
                    "we": 0.05,
                },
                "cc": {
                    "a_s": 0.179367259,
                    "b_s": 0.0118889215,
                    "c_s1": 0.0563691592,
                    "c_s2": 0.13290157,
                    "a_f": 3.11,
                    "b_f": -0.68,
                    "c_f": 2.41,
                    "alpha_mod": 1.0
                },
                "turbopark": {
                    "A": 0.04,
                    "sigma_max_rel": 4.0
                }
            },
            "wake_turbulence_parameters": {
                "crespo_hernandez": {
                    "initial": 0.1,
                    "constant": 0.5,
                    "ai": 0.8,
                    "downstream": -0.32
                },
            },
            "enable_secondary_steering": False,
            "enable_yaw_added_recovery": False,
            "enable_transverse_velocities": False,
        }

        self.floris = {
            "farm": self.farm,
            "flow_field": self.flow_field,
            "wake": self.wake,
            "solver": {
                "type": "turbine_grid",
                "turbine_grid_points": 5,
            },
            "logging": {
                "console": {"enable": True, "level": 1},
                "file": {"enable": False, "level": 1},
            },
            "name": "conftest",
            "description": "Inputs used for testing",
            "floris_version": "v3.0.0",
        }
