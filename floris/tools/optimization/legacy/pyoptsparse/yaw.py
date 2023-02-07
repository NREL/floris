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
from scipy.stats import norm

from ...visualization import visualize_cut_plane


class Yaw:
    """
    Class that performs yaw optimization for a single set of
    inflow conditions. Intended to be used together with an object of the
    :py:class`floris.tools.optimization.optimization.Optimization` class.

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`):
            Interface from FLORIS to the tools package.
        minimum_yaw_angle (float, optional): Minimum constraint on
            yaw. Defaults to None.
        maximum_yaw_angle (float, optional): Maximum constraint on
            yaw. Defaults to None.
        x0 (iterable, optional): The initial yaw conditions.
            Defaults to None. Initializes to the current turbine
            yaw settings.
        include_unc (bool): If True, uncertainty in wind direction
            and/or yaw position is included when determining wind farm power.
            Uncertainty is included by computing the mean wind farm power for
            a distribution of wind direction and yaw position deviations from
            the original wind direction and yaw angles. Defaults to False.
        unc_pmfs (dictionary, optional): A dictionary containing optional
            probability mass functions describing the distribution of wind
            direction and yaw position deviations when wind direction and/or
            yaw position uncertainty is included in the power calculations.
            Contains the following key-value pairs:

            -   **wd_unc**: A numpy array containing wind direction deviations
                from the original wind direction.
            -   **wd_unc_pmf**: A numpy array containing the probability of
                each wind direction deviation in **wd_unc** occuring.
            -   **yaw_unc**: A numpy array containing yaw angle deviations
                from the original yaw angles.
            -   **yaw_unc_pmf**: A numpy array containing the probability of
                each yaw angle deviation in **yaw_unc** occuring.

            Defaults to None, in which case default PMFs are calculated using
            values provided in **unc_options**.
        unc_options (disctionary, optional): A dictionary containing values used
            to create normally-distributed, zero-mean probability mass functions
            describing the distribution of wind direction and yaw position
            deviations when wind direction and/or yaw position uncertainty is
            included. This argument is only used when **unc_pmfs** is None and
            contains the following key-value pairs:

            -   **std_wd**: A float containing the standard deviation of the wind
                    direction deviations from the original wind direction.
            -   **std_yaw**: A float containing the standard deviation of the yaw
                    angle deviations from the original yaw angles.
            -   **pmf_res**: A float containing the resolution in degrees of the
                    wind direction and yaw angle PMFs.
            -   **pdf_cutoff**: A float containing the cumulative distribution
                function value at which the tails of the PMFs are truncated.

            Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75,
            'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
        wdir (float, optional): Wind direction to use for optimization. Defaults
            to None. Initializes to current wind direction in floris.
        wspd (float, optional): Wind speed to use for optimization. Defaults
            to None. Initializes to current wind direction in floris.

    Returns:
        Yaw: An instantiated Yaw object.
    """

    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        x0=None,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        wdir=None,
        wspd=None,
    ):
        """
        Instantiate Yaw object and parameter values.
        """
        self.fi = fi
        self.minimum_yaw_angle = minimum_yaw_angle
        self.maximum_yaw_angle = maximum_yaw_angle

        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [
                turbine.yaw_angle
                for turbine in self.fi.floris.farm.turbine_map.turbines
            ]

        self.include_unc = include_unc
        self.unc_pmfs = unc_pmfs
        if self.include_unc & (self.unc_pmfs is None):
            self.unc_pmfs = calc_unc_pmfs(self.unc_pmfs)

        if wdir is not None:
            self.wdir = wdir
        else:
            self.wdir = self.fi.floris.farm.flow_field.wind_direction
        if wspd is not None:
            self.wspd = wspd
        else:
            self.wspd = self.fi.floris.farm.flow_field.wind_speed

        self.fi.reinitialize_flow_field(wind_speed=self.wspd, wind_direction=self.wdir)

    def __str__(self):
        return "yaw"

    ###########################################################################
    # Required private optimization methods
    ###########################################################################

    def reinitialize(self):
        pass

    def obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Reinitialize with wind speed and direction
        self.fi.reinitialize_flow_field(wind_speed=self.wspd, wind_direction=self.wdir)

        # Compute the objective function
        funcs = {}
        funcs["obj"] = -1 * self.fi.get_farm_power_for_yaw_angle(self.yaw) / 1e0

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs)

        fail = False
        return funcs, fail

    def parse_opt_vars(self, varDict):
        self.yaw = varDict["yaw"]

    def parse_sol_vars(self, sol):
        self.yaw = list(sol.getDVs().values())[0]

    def add_var_group(self, optProb):
        optProb.addVarGroup(
            "yaw",
            self.nturbs,
            type="c",
            lower=self.minimum_yaw_angle,
            upper=self.maximum_yaw_angle,
            value=self.x0,
        )

        return optProb

    def add_con_group(self, optProb):
        # no constraints defined
        return optProb

    def compute_cons(self, funcs):
        # no constraints defined
        return funcs

    ###########################################################################
    # User-defined methods
    ###########################################################################

    def plot_yaw_opt_results(self, sol):
        """
        Method to plot the wind farm with optimal yaw offsets
        """
        yaw = sol.getDVs()["yaw"]

        # Assign yaw angles to turbines and calculate wake
        self.fi.calculate_wake(yaw_angles=yaw)

        # Initialize the horizontal cut
        horizontal_plane = self.fi.calculate_horizontal_plane(x_resolution=400, y_resolution=100)

        # Plot and show
        fig, ax = plt.subplots()
        visualize_cut_plane(horizontal_plane, ax=ax)
        ax.set_title(
            "Optimal Yaw Offsets for U = "
            + str(self.wspd[0])
            + " m/s, Wind Direction = "
            + str(self.wdir[0])
            + "$^\\circ$"
        )

        plt.show()

    def print_power_gain(self, sol):
        """
        Method to print the power gain from wake steering with optimal yaw offsets
        """
        yaw = sol.getDVs()["yaw"]

        self.fi.calculate_wake(yaw_angles=0.0)
        power_baseline = self.fi.get_farm_power()

        self.fi.calculate_wake(yaw_angles=yaw)
        power_opt = self.fi.get_farm_power()

        pct_gain = 100.0 * (power_opt - power_baseline) / power_baseline

        print("==========================================")
        print("Baseline Power = %.1f kW" % (power_baseline / 1e3))
        print("Optimal Power = %.1f kW" % (power_opt / 1e3))
        print("Total Power Gain = %.1f%%" % pct_gain)
        print("==========================================")

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = len(self.fi.floris.farm.turbines)
        return self._nturbs


def calc_unc_pmfs(unc_options=None):
    """
    Calculates normally-distributed probability mass functions describing the
    distribution of wind direction and yaw position deviations when wind direction
    and/or yaw position uncertainty are included in power calculations.

    Args:
        unc_options (dictionary, optional): A dictionary containing values used
                to create normally-distributed, zero-mean probability mass functions
                describing the distribution of wind direction and yaw position
                deviations when wind direction and/or yaw position uncertainty is
                included. This argument is only used when **unc_pmfs** is None and
                contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of the wind
                        direction deviations from the original wind direction.
                -   **std_yaw**: A float containing the standard deviation of the yaw
                        angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees of the
                        wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative distribution
                    function value at which the tails of the PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75,
                'pmf_res': 1.0, 'pdf_cutoff': 0.995}.

    Returns:
        [dictionary]: A dictionary containing
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc**: A numpy array containing wind direction deviations
                    from the original wind direction.
                -   **wd_unc_pmf**: A numpy array containing the probability of
                    each wind direction deviation in **wd_unc** occuring.
                -   **yaw_unc**: A numpy array containing yaw angle deviations
                    from the original yaw angles.
                -   **yaw_unc_pmf**: A numpy array containing the probability of
                    each yaw angle deviation in **yaw_unc** occuring.

    """

    if unc_options is None:
        unc_options = {
            "std_wd": 4.95,
            "std_yaw": 1.75,
            "pmf_res": 1.0,
            "pdf_cutoff": 0.995,
        }

    # create normally distributed wd and yaw uncertainty pmfs
    if unc_options["std_wd"] > 0:
        wd_bnd = int(
            np.ceil(
                norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_wd"])
                / unc_options["pmf_res"]
            )
        )
        wd_unc = np.linspace(
            -1 * wd_bnd * unc_options["pmf_res"],
            wd_bnd * unc_options["pmf_res"],
            2 * wd_bnd + 1,
        )
        wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options["std_wd"])
        wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf)  # normalize so sum = 1.0
    else:
        wd_unc = np.zeros(1)
        wd_unc_pmf = np.ones(1)

    if unc_options["std_yaw"] > 0:
        yaw_bnd = int(
            np.ceil(
                norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_yaw"])
                / unc_options["pmf_res"]
            )
        )
        yaw_unc = np.linspace(
            -1 * yaw_bnd * unc_options["pmf_res"],
            yaw_bnd * unc_options["pmf_res"],
            2 * yaw_bnd + 1,
        )
        yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options["std_yaw"])
        yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf)  # normalize so sum = 1.0
    else:
        yaw_unc = np.zeros(1)
        yaw_unc_pmf = np.ones(1)

    return {
        "wd_unc": wd_unc,
        "wd_unc_pmf": wd_unc_pmf,
        "yaw_unc": yaw_unc,
        "yaw_unc_pmf": yaw_unc_pmf,
    }
