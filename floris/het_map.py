from __future__ import annotations

import inspect
from abc import abstractmethod
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from floris.type_dec import NDArrayFloat


class HetMap:
    """
    Class for handling heterogeneous inflow configurations when defined by wind direction
      and wind speed.
    Args:
        heterogeneous_inflow_config_by_wd (dict): A dictionary containing the following
            keys:
            * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
            * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).
            * 'speed_multipliers': A 2D NumPy array (size num_wd (or num_ws) x num_points)
                of speed multipliers.  If neither wind_directions nor wind_speeds are
                defined, then this should be a single row array
            * 'wind_directions': A 1D NumPy array (size num_wd) of wind directions (degrees).
                Optional.
            * 'wind_speeds': A 1D NumPy array (size num_ws) of wind speeds (m/s). Optional.

    Notes:
        * If 'wind_directions' and 'wind_speeds' are both defined, then they must be the same length
            and equal the length of the 0th dimension of 'speed_multipliers'.

    """

    def __init__(
        self,
        heterogeneous_inflow_config_by_wd: dict,
    ):
        # Check the dictionary contains x, y, and speed_multipliers
        if not isinstance(heterogeneous_inflow_config_by_wd, dict):
            raise TypeError("heterogeneous_inflow_config_by_wd must be a dictionary")
        if "x" not in heterogeneous_inflow_config_by_wd:
            raise ValueError("heterogeneous_inflow_config_by_wd must contain a key 'x'")
        if "y" not in heterogeneous_inflow_config_by_wd:
            raise ValueError("heterogeneous_inflow_config_by_wd must contain a key 'y'")
        if "speed_multipliers" not in heterogeneous_inflow_config_by_wd:
            raise ValueError(
                "heterogeneous_inflow_config_by_wd must contain a key 'speed_multipliers'"
            )

        # Save the values
        self.x = np.array(heterogeneous_inflow_config_by_wd["x"])
        self.y = np.array(heterogeneous_inflow_config_by_wd["y"])
        self.speed_multipliers = np.array(heterogeneous_inflow_config_by_wd["speed_multipliers"])

        # Check that the length of the 1st dimension of speed_multipliers is the
        # same as the length of both x and y
        if len(self.x) != self.speed_multipliers.shape[1]:
            raise ValueError(
                "The length of x must equal the 1th dimension of speed_multipliers"
                "Within the heterogeneous_inflow_config_by_wd dictionary"
            )
        if len(self.y) != self.speed_multipliers.shape[1]:
            raise ValueError(
                "The length of y must equal the 1th dimension of speed_multipliers"
                "Within the heterogeneous_inflow_config_by_wd dictionary"
            )

        # If wind_directions is a defined element of the dictionary, then save it
        if "wind_directions" in heterogeneous_inflow_config_by_wd:
            self.wind_directions = np.array(heterogeneous_inflow_config_by_wd["wind_directions"])

            # Check that length of wind_directions is the same as the length of the 0th
            # dimension of speed_multipliers
            if len(self.wind_directions) != self.speed_multipliers.shape[0]:
                raise ValueError(
                    "The length of wind_directions must equal "
                    "the 0th dimension of speed_multipliers"
                    "Within the heterogeneous_inflow_config_by_wd dictionary"
                )
        else:
            self.wind_directions = None

        # If wind_speeds is a defined element of the dictionary, then save it
        if "wind_speeds" in heterogeneous_inflow_config_by_wd:
            self.wind_speeds = np.array(heterogeneous_inflow_config_by_wd["wind_speeds"])

            # Check that length of wind_speeds is the same as the length of the 1th
            # dimension of speed_multipliers
            if len(self.wind_speeds) != self.speed_multipliers.shape[0]:
                raise ValueError(
                    "The length of wind_speeds must equal the 1th dimension of speed_multipliers"
                    "Within the heterogeneous_inflow_config_by_wd dictionary"
                )
        else:
            self.wind_speeds = None

        # If both wind_directions and wind_speeds are None, then speed_multipliers should be
        # length 1 in 0th dimension
        if self.wind_speeds is None and self.wind_directions is None:
            if self.speed_multipliers.shape[0] != 1:
                raise ValueError(
                    "If both wind_speeds and wind_directions are None, then speed_multipliers"
                    "should be length 1 in 0th dimension"
                )

        # If both wind_directions and wind_speeds are not None, then make sure each row
        # of a matrix where wind directions and wind speeds are the columns is unique
        if self.wind_speeds is not None and self.wind_directions is not None:
            if len(
                np.unique(np.column_stack((self.wind_directions, self.wind_speeds)), axis=0)
            ) != len(self.wind_directions):
                raise ValueError(
                    "Each row of a matrix where wind directions and wind speeds are the columns"
                    "should be unique"
                )

        # Save the heterogeneous_inflow_config_by_wd dictionary
        # self.heterogeneous_inflow_config_by_wd = heterogeneous_inflow_config_by_wd

    def get_heterogeneous_inflow_config(
        self,
        wind_directions,
        wind_speeds,
    ):
        # Check the wind_directions and wind_speeds are either lists or numpy arrays,
        # and are the same length
        if not isinstance(wind_directions, (list, np.ndarray)):
            raise TypeError("wind_directions must be a list or numpy array")
        if not isinstance(wind_speeds, (list, np.ndarray)):
            raise TypeError("wind_speeds must be a list or numpy array")
        if len(wind_directions) != len(wind_speeds):
            raise ValueError("wind_directions and wind_speeds must be the same length")

        # Select for wind direction first
        if self.wind_directions is not None:
            angle_diffs = np.abs(wind_directions[:, None] - self.wind_directions)
            min_angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)

            # If wind_speeds is none, can return the value in each case
            if self.wind_speeds is None:
                closest_wd_indices = np.argmin(min_angle_diffs, axis=1)

                # Construct the output array using the calculated indices
                speed_multipliers_by_findex = self.speed_multipliers[closest_wd_indices]

            # Need to loop over cases and match by wind speed
            else:
                speed_diffs = np.abs(wind_speeds[:, None] - self.wind_speeds)

                # Initialize the output array
                speed_multipliers_by_findex = np.zeros((len(wind_directions), len(self.x)))

                # Loop over each wind direction
                for i in range(len(wind_directions)):
                    # Find all the indices in the ith row of min_angle_diffs
                    # that are equal to the minimum value
                    closest_wd_indices = np.where(min_angle_diffs[i] == min_angle_diffs[i].min())[0]

                    # Find the index of the minimum value in the ith row of speed_diffs
                    # conditions on that index being in closest_wd_indices
                    closest_ws_index = np.argmin(speed_diffs[i, closest_wd_indices])

                    # Construct the output array using the calculated indices
                    speed_multipliers_by_findex[i] = self.speed_multipliers[
                        closest_wd_indices[closest_ws_index]
                    ]

        # If wind speeds are defined without wind direction
        elif self.wind_speeds is not None:
            speed_diffs = np.abs(wind_speeds[:, None] - self.wind_speeds)
            closest_ws_indices = np.argmin(speed_diffs, axis=1)

            # Construct the output array using the calculated indices
            speed_multipliers_by_findex = self.speed_multipliers[closest_ws_indices]

        # Else if both are None, then speed_multipliers should be length 1 in 0th
        # dimension and so just 1 row
        # repeat this row until length of wind_directions
        else:
            speed_multipliers_by_findex = np.repeat(
                self.speed_multipliers, len(wind_directions), axis=0
            )

        # Return heterogeneous_inflow_config
        return {
            "x": self.x,
            "y": self.y,
            "speed_multipliers": speed_multipliers_by_findex,
        }
