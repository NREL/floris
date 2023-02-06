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

from ..utilities import wrap_180, wrap_360


def log_law_interpolate(z_test, z_ref, v_ref, roughness=0.03):
    """
    Interpolate wind speed assuming a log-law profile.

    Args:
        z_test (float): height of interest for wind speed estimate.
        z_ref (float): reference height.
        v_ref (float): reference velocity.
        roughness (float, optional): Effective roughness length.
            Defaults to 0.03.

    Returns:
        v_test (np.float): interpolated wind speed at z_test.
    """
    return v_ref * np.log(z_test / roughness) / np.log(z_ref / roughness)


def determine_rews_weights(R, HH, heights_in):
    """
    Weighting for rotor-equivalent wind speed (REWS).

    Args:
        R (float): rotor diameter.
        HH (float): hub height.
        heights_in (iterable): heights of interest.

    Returns:
        weights_return (list): list of weighting values for REWS.
    """
    # Remove any heights not in range of the rotor
    heights = [h for h in heights_in if ((h >= HH - R) and (h <= HH + R))]
    num_heights = len(heights)

    # Determine the zone interfaces
    zone_boundaries = np.zeros(num_heights + 1)
    zone_boundaries[0] = HH - R
    zone_boundaries[-1] = HH + R
    for i in range(1, num_heights):
        zone_boundaries[i] = (heights[i] - heights[i - 1]) / 2.0 + heights[i - 1]
    zone_interfaces = zone_boundaries[1:-1]

    # Next find the central angles for each interace
    h = zone_interfaces - HH
    alpha = np.arcsin(h / R)
    C = np.pi - 2 * alpha
    A = ((R ** 2) / 2) * (C - np.sin(C))
    A = [np.pi * R ** 2] + list(A)
    for i in range(num_heights - 1):
        A[i] = A[i] - A[i + 1]
    weights = A

    # normalize
    weights = weights / np.sum(weights)

    # Now re-pad weights to include heights that were initally cropped
    weight_dict = dict(zip(heights, weights))
    weights_return = [weight_dict.get(h, 0.0) for h in heights_in]

    return weights_return


def rews_from_df(df, columns_in, weights, rews_name, circular=False):
    """
    Estimate the rotor-equivalent wind speed (REWS) from wind speed.

    Args:
        df (pd.DataFrame): DataFrame containing flow information
        columns_in (list): columns to include estimate of REWS.
        weights (iterable): weighting values for REWS.
        rews_name (str): column name for REWS output.
        circular (bool, optional): flag to consider REWS azimuthally.
            Defaults to False.

    Returns:
        df (pd.DataFrame): updated dataframe with REWS column.
    """
    # Ensure numpy array
    weights = np.array(weights)

    # Get the data
    data_matrix = df[columns_in].values

    if not circular:
        df[rews_name] = compute_rews(data_matrix, weights)
    else:
        cos_vals = compute_rews(np.cos(np.deg2rad(data_matrix)), weights)
        sin_vals = compute_rews(np.sin(np.deg2rad(data_matrix)), weights)
        df[rews_name] = wrap_360(np.rad2deg(np.arctan2(sin_vals, cos_vals)))

    return df


def compute_rews(data_matrix, weights):
    """
    Calculation method for REWS from wind speed and weighting values.

    Args:
        data_matrix (np.array): wind speed data
        weights (np.array): weighting values for REWS.

    Returns:
        REWS (float): rotor-equivalent wind speed.
    """
    return np.sum(data_matrix * weights, axis=1)
