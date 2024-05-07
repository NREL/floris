"""Example: Using Het Map

...
"""

import matplotlib.pyplot as plt
import numpy as np

from floris.heterogeneous_map import HeterogeneousMap


heterogeneous_inflow_config_by_wd = {
    "x": np.array([0.0, 0.0, 500.0, 500.0]),
    "y": np.array([0.0, 500.0, 0.0, 500.0 ]),
    "speed_multipliers": np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.25, 1.0, 1.25],
            [1.0, 1.0, 1.0, 1.25],
            [1.0, 1.0, 1.0, 1.0],
        ]
    ),
    "wind_directions": np.array([0.0, 0.0, 90.0, 90.0]),
    "wind_speeds": np.array([5.0, 15.0, 5.0, 15.0]),
}

hm = HeterogeneousMap(heterogeneous_inflow_config_by_wd)
