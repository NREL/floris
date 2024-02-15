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


# Defines a bunch of tools for plotting and manipulating
# layouts for quick visualizations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def get_wake_direction(x_i, y_i, x_j, y_j):
    """
    Calculates the compass direction where turbine i wakes turbine j

    Args:
        x_i: x-coordinate of the starting point
        y_i: y-coordinate of the starting point
        x_j: x-coordinate of the ending point
        y_j: y-coordinate of the ending point

    Returns:
        wake_direction (float): Angle in degrees, when turbine i wakes turbine j
    """

    dx = x_j - x_i
    dy = y_j - y_i

    angle_rad = np.arctan2(dy, dx)
    angle_deg = 270 - np.rad2deg(angle_rad)

    # Adjust for "from" direction (add 180 degrees) and wrap within 0-360
    wind_direction = angle_deg % 360

    return wind_direction
