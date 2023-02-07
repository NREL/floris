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


import matplotlib.pyplot as plt
import numpy as np


def derive_downstream_turbines(fi, wind_direction, wake_slope=0.30, plot_lines=False):
    """Determine which turbines have no effect on other turbines in the
    farm, i.e., which turbines have wakes that do not impact the other
    turbines in the farm. This allows the user to exclude these turbines
    from a control setpoint optimization, for example. This function
    assumes a very simplified wake function where the wakes are assumed
    to have a linearly diverging profile. In comparisons with the FLORIS
    GCH model, the wake_slope matches well with the FLORIS' wake profiles
    for a value of wake_slope = 0.5 * turbulence_intensity, where
    turbulence_intensity is an input to the FLORIS model at the default
    GCH parameterization. Note that does not include wind direction variability.
    To be conservative, the user is recommended to use the rule of thumb:
    `wake_slope = turbulence_intensity`. Hence, the default value for
    `wake_slope=0.30` should be conservative for turbulence intensities up to
    0.30 and is likely to provide valid estimates of which turbines are
    downstream until a turbulence intensity of 0.50. This simple model saves
    time compared to FLORIS.

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        wind_direction (float): The wind direction in the FLORIS frame
        of reference for which the downstream turbines are to be determined.
        wake_slope (float, optional): linear slope of the wake (dy/dx)
        plot_lines (bool, optional): Enable plotting wakes/turbines.
        Defaults to False.

    Returns:
        turbs_downstream (iterable): A list containing the turbine
        numbers that have a wake that does not affect any other
        turbine inside the farm.
    """

    # Get farm layout
    x = fi.layout_x
    y = fi.layout_y
    D = np.array([t.rotor_diameter for t in fi.floris.farm.turbines])
    n_turbs = len(x)

    # Rotate farm and determine freestream/waked turbines
    is_downstream = [False for _ in range(n_turbs)]
    x_rot = (
        np.cos((wind_direction - 270.0) * np.pi / 180.0) * x
        - np.sin((wind_direction - 270.0) * np.pi / 180.0) * y
    )
    y_rot = (
        np.sin((wind_direction - 270.0) * np.pi / 180.0) * x
        + np.cos((wind_direction - 270.0) * np.pi / 180.0) * y
    )

    if plot_lines:
        fig, ax = plt.subplots()
        for ii in range(n_turbs):
            ax.plot(
                x_rot[ii] * np.ones(2),
                [y_rot[ii] - D[ii] / 2, y_rot[ii] + D[ii] / 2],
                "k",
            )
        for ii in range(n_turbs):
            ax.text(x_rot[ii], y_rot[ii], "T%03d" % ii)
        ax.axis("equal")

    srt = np.argsort(x_rot)
    x_rot_srt = x_rot[srt]
    y_rot_srt = y_rot[srt]
    for ii in range(n_turbs):
        x0 = x_rot_srt[ii]
        y0 = y_rot_srt[ii]

        def wake_profile_ub_turbii(x):
            y = (y0 + D[ii]) + (x - x0) * wake_slope
            if isinstance(y, (float, np.float64, np.float32)):
                if x < (x0 + 0.01):
                    y = -np.Inf
            else:
                y[x < x0 + 0.01] = -np.Inf
            return y

        def wake_profile_lb_turbii(x):
            y = (y0 - D[ii]) - (x - x0) * wake_slope
            if isinstance(y, (float, np.float64, np.float32)):
                if x < (x0 + 0.01):
                    y = -np.Inf
            else:
                y[x < x0 + 0.01] = -np.Inf
            return y

        def determine_if_in_wake(xt, yt):
            return (yt < wake_profile_ub_turbii(xt)) & (yt > wake_profile_lb_turbii(xt))

        is_downstream[ii] = not any(
            [
                determine_if_in_wake(x_rot_srt[iii], y_rot_srt[iii])
                for iii in range(n_turbs)
            ]
        )

        if plot_lines:
            x1 = np.max(x_rot_srt) + 500.0
            ax.fill_between(
                [x0, x1, x1, x0],
                [
                    wake_profile_ub_turbii(x0 + 0.02),
                    wake_profile_ub_turbii(x1),
                    wake_profile_lb_turbii(x1),
                    wake_profile_lb_turbii(x0 + 0.02),
                ],
                alpha=0.1,
                color="k",
                edgecolor=None,
            )

    usrt = np.argsort(srt)
    is_downstream = [is_downstream[i] for i in usrt]
    turbs_downstream = list(np.where(is_downstream)[0])

    if plot_lines:
        ax.set_title("wind_direction = %03d" % wind_direction)
        ax.set_xlim([np.min(x_rot) - 500.0, x1])
        ax.set_ylim([np.min(y_rot) - 500.0, np.max(y_rot) + 500.0])
        ax.plot(
            x_rot[turbs_downstream], y_rot[turbs_downstream], "o", color="green",
        )

    return turbs_downstream
