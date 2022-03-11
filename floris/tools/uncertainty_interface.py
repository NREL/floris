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


import copy
import numpy as np
from scipy.stats import norm

from floris.logging_manager import LoggerBase
from floris.utilities import wrap_360


class UncertaintyInterface(LoggerBase):

    def __init__(self, fi, unc_options=None, unc_pmfs=None):
        """A wrapper around the nominal floris_interface class that adds
        uncertainty to the floris evaluations. One can specify a probability
        distribution function (pdf) for the ambient wind direction, and for
        the turbine yaw angles. Unless the exact pdf is specified manually
        using the option 'unc_pmfs', a Gaussian distribution function will be
        assumed.

        Args:
            fi (FlorisInterface): The FlorisInterface object.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw':
                1.75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
        """

        if (unc_options is None) & (unc_pmfs is None):
            # Default options:
            unc_options = {
                "std_wd": 3.0,  # Standard deviation for inflow wind direction (deg)
                "std_yaw": 0.0,  # Standard deviation in turbine yaw angle (deg)
                "pmf_res": 1.0,  # Resolution over which to calculate angles (deg)
                "pdf_cutoff": 0.995,  # Probability density function cut-off (-)
            }

        # Save floris object to self and initialize uncertainty pdfs
        self.fi = fi
        self.reinitialize_uncertainty(unc_options=unc_options, unc_pmfs=unc_pmfs)

    # Private methods

    def _generate_pdfs_from_dict(self):
        """Generates the uncertainty probability distributions from a
        dictionary only describing the wd_std and yaw_std, and discretization
        resolution.
        """

        wd_unc = np.zeros(1)
        wd_unc_pmf = np.ones(1)
        yaw_unc = np.zeros(1)
        yaw_unc_pmf = np.ones(1)

        # create normally distributed wd and yaw uncertaitny pmfs if appropriate
        unc_options = self.unc_options
        if unc_options["std_wd"] > 0:
            wd_bnd = int(np.ceil(norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_wd"]) / unc_options["pmf_res"]))
            bound = wd_bnd * unc_options["pmf_res"]
            wd_unc = np.linspace(-1 * bound, bound, 2 * wd_bnd + 1)
            wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options["std_wd"])
            wd_unc_pmf /= np.sum(wd_unc_pmf)  # normalize so sum = 1.0

        if unc_options["std_yaw"] > 0:
            yaw_bnd = int(
                np.ceil(norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_yaw"]) / unc_options["pmf_res"])
            )
            bound = yaw_bnd * unc_options["pmf_res"]
            yaw_unc = np.linspace(-1 * bound, bound, 2 * yaw_bnd + 1)
            yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options["std_yaw"])
            yaw_unc_pmf /= np.sum(yaw_unc_pmf)  # normalize so sum = 1.0

        unc_pmfs = {
            "wd_unc": wd_unc,
            "wd_unc_pmf": wd_unc_pmf,
            "yaw_unc": yaw_unc,
            "yaw_unc_pmf": yaw_unc_pmf,
        }

        # Save to self
        self.unc_pmfs = unc_pmfs

    def _expand_wind_directions_and_yaw_angles(self):
        """Expands the nominal wind directions and yaw angles to the full set
        of conditions that need to be evaluated for the probablistic
        calculation of the floris solutions. This produces the np.NDArrays
        "wd_array_probablistic" and "yaw_angles_probablistic", with shapes:
            (
                num_yaw_angle_pdf_points_to_evaluate,
                num_wind_direction_pdf_points_to_evaluate,
                num_nominal_wind_directions,
            )
            and
            (
                num_yaw_angle_pdf_points_to_evaluate,
                num_wind_direction_pdf_points_to_evaluate,
                num_nominal_wind_directions,
                num_turbines
            ),
            respectively.
        """

        # First initialize unc_pmfs from self
        unc_pmfs = self.unc_pmfs

        # We first save the nominal settings, since we will be overwriting
        # the floris wind conditions and yaw angles to include all
        # probablistic conditions.
        wd_array_nominal = self.fi.floris.flow_field.wind_directions
        yaw_angles_nominal = self.fi.floris.farm.yaw_angles

        # Expand wind direction and yaw angle array into the direction
        # of uncertainty over the ambient wind direction.
        wd_array_probablistic = np.vstack(
            [np.expand_dims(wd_array_nominal, axis=0) + dy
            for dy in unc_pmfs["wd_unc"]]
        )
        yaw_angles_probablistic = np.vstack(
            [np.expand_dims(yaw_angles_nominal, axis=0) - dy
            for dy in unc_pmfs["wd_unc"]]
        )

        # Now futher expand wind direction and yaw angle array into
        # the direction of uncertainty of the turbine yaw angles.
        wd_array_probablistic = np.vstack(
            [np.expand_dims(wd_array_probablistic, axis=0)
            for _ in unc_pmfs["yaw_unc"]]
        )
        yaw_angles_probablistic = np.vstack(
            [np.expand_dims(yaw_angles_probablistic, axis=0) + dy
            for dy in unc_pmfs["yaw_unc"]]
        )

        self.wd_array_probablistic = wd_array_probablistic
        self.yaw_angles_probablistic = yaw_angles_probablistic


    def _reassign_yaw_angles(self, yaw_angles=None):
        # Overwrite the yaw angles in the FlorisInterface object
        if yaw_angles is not None:
            self.fi.floris.farm.yaw_angles = yaw_angles

    # Public methods

    def copy(self):
        """Create an independent copy of the current UncertaintyInterface
        object"""
        fi_unc_copy = copy.deepcopy(self)
        fi_unc_copy.fi = self.fi.copy()
        return fi_unc_copy

    def reinitialize_uncertainty(self, unc_options=None, unc_pmfs=None):
        """Reinitialize the wind direction and yaw angle probability
        distributions used in evaluating FLORIS. Must either specify
        'unc_options', in which case distributions are calculated assuming
        a Gaussian distribution, or `unc_pmfs` must be specified directly
        assigning the probability distribution functions.

        Args:
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None.

            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None.
        """

        # Check inputs
        if ((unc_options is not None) and (unc_pmfs is not None)):
            self.logger.error(
                "Must specify either 'unc_options' or 'unc_pmfs', not both."
            )

        # Assign uncertainty probability distributions
        if unc_options is not None:
            self.unc_options = unc_options
            self._generate_pdfs_from_dict()
        
        if unc_pmfs is not None:
            self.unc_pmfs = unc_pmfs

    def reinitialize(
        self,
        wind_speeds=None,
        wind_directions=None,
        wind_shear=None,
        wind_veer=None,
        reference_wind_height=None,
        turbulence_intensity=None,
        air_density=None,
        layout=None,
        turbine_type=None,
        solver_settings=None,
    ):
        """Pass to the FlorisInterface reinitialize function. To allow users
        to directly replace a FlorisInterface object with this
        UncertaintyInterface object, this function is required."""

        # Just passes arguments to the floris object
        self.fi.reinitialize(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            reference_wind_height=reference_wind_height,
            turbulence_intensity=turbulence_intensity,
            air_density=air_density,
            layout=layout,
            turbine_type=turbine_type,
            solver_settings=solver_settings,
        )

    def calculate_wake(self, yaw_angles=None):
        """Replaces the 'calculate_wake' function in the FlorisInterface
        object. Fundamentally, this function only overwrites the nominal
        yaw angles in the FlorisInterface object. The actual wake calculations
        are performed once 'get_turbine_powers' or 'get_farm_powers' is
        called. However, to allow users to directly replace a FlorisInterface
        object with this UncertaintyInterface object, this function is
        required.

        Args:
            yaw_angles: NDArrayFloat | list[float] | None = None,
        """
        self._reassign_yaw_angles(yaw_angles)

    def get_turbine_powers(self, no_wake=False):
        """Calculates the probability-weighted power production of each
        turbine in the wind farm.

        Args:
            no_wake (bool, optional): disable the wakes in the flow model.
            This can be useful to determine the (probablistic) power
            production of the farm in the artificial scenario where there
            would never be any wake losses. Defaults to False.

        Returns:
            NDArrayFloat: Power production of all turbines in the wind farm.
            This array has the shape (num_wind_directions, num_wind_speeds,
            num_turbines).
        """

        # To include uncertainty, we expand the dimensionality
        # of the problem along the wind direction pdf and/or yaw angle
        # pdf. We make use of the vectorization of FLORIS to
        # evaluate all conditions in a single call, rather than in
        # loops. Therefore, the effective number of wind conditions and
        # yaw angle combinations we evaluate expands.
        unc_pmfs = self.unc_pmfs
        self._expand_wind_directions_and_yaw_angles()

        # Get dimensions of nominal conditions
        wd_array_nominal = self.fi.floris.flow_field.wind_directions
        num_wd = self.fi.floris.flow_field.n_wind_directions
        num_ws = self.fi.floris.flow_field.n_wind_speeds
        num_wd_unc = len(unc_pmfs["wd_unc"])
        num_yaw_unc = len(unc_pmfs["yaw_unc"])
        num_turbines = self.fi.floris.farm.n_turbines

        # Format into conventional floris format by reshaping
        wd_array_probablistic = np.reshape(self.wd_array_probablistic, -1)
        yaw_angles_probablistic = np.reshape(
            self.yaw_angles_probablistic, (-1, num_ws, num_turbines)
        )

        # Wrap wind direction array around 360 deg
        wd_array_probablistic = wrap_360(wd_array_probablistic)

        # Find minimal set of solutions to evaluate
        wd_exp = np.tile(wd_array_probablistic, (1, num_ws, 1)).T
        x_exp = np.append(yaw_angles_probablistic, wd_exp, axis=2)
        _, id_unq, id_unq_rev = np.unique(
            x_exp,
            axis=0,
            return_index=True,
            return_inverse=True
        )
        wd_array_probablistic_min = wd_array_probablistic[id_unq]
        yaw_angles_probablistic_min = yaw_angles_probablistic[id_unq, :, :]

        # Evaluate floris for minimal probablistic set
        self.fi.reinitialize(wind_directions=wd_array_probablistic_min)
        self.fi.calculate_wake(
            yaw_angles=yaw_angles_probablistic_min,
            no_wake=no_wake
        )

        # Retrieve all power productions using the nominal call
        turbine_powers = self.fi.get_turbine_powers()
        self.fi.reinitialize(wind_directions=wd_array_nominal)

        # Reshape solutions back to full set
        power_probablistic = turbine_powers[id_unq_rev, :]
        power_probablistic = np.reshape(
            power_probablistic, 
            (num_yaw_unc, num_wd_unc, num_wd, num_ws, num_turbines)
        )

        # Calculate probability weighing terms
        wd_weighing = np.tile(unc_pmfs["wd_unc_pmf"], (num_yaw_unc, 1)).T
        wd_weighing = np.tile(wd_weighing, (num_ws, num_wd, 1, 1)).T
        yaw_weighing = np.tile(unc_pmfs["yaw_unc_pmf"], (num_wd_unc, 1))
        yaw_weighing = np.tile(yaw_weighing, (num_ws, num_wd, 1, 1)).T

        # Now copy over to each turbine
        wd_weighing = np.repeat(
            np.expand_dims(wd_weighing, axis=4),
            num_turbines,
            axis=4
        )
        yaw_weighing = np.repeat(
            np.expand_dims(yaw_weighing, axis=4),
            num_turbines,
            axis=4
        )
        W = np.multiply(wd_weighing, yaw_weighing)

        # Now apply probability distribution weighing to get turbine powers
        return np.sum(W * power_probablistic, axis=(0, 1))

    def get_farm_power(self, no_wake=False):
        """Calculates the probability-weighted power production of the
        collective of all turbines in the farm, for each wind direction
        and wind speed specified.

        Args:
            no_wake (bool, optional): disable the wakes in the flow model.
            This can be useful to determine the (probablistic) power
            production of the farm in the artificial scenario where there
            would never be any wake losses. Defaults to False.

        Returns:
            NDArrayFloat: Expectation of power production of the wind farm.
            This array has the shape (num_wind_directions, num_wind_speeds).
        """
        turbine_powers = self.get_turbine_powers(no_wake=no_wake)
        return np.sum(turbine_powers, axis=2)

    # Define getter functions that just pass information from FlorisInterface
    @property
    def floris(self):
        return self.fi.floris

    @property
    def layout_x(self):
        return self.fi.layout_x

    @property
    def layout_y(self):
        return self.fi.layout_y
