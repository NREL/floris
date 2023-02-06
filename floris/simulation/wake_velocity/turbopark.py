# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io
from attrs import define, field
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator

from floris.simulation import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class TurbOParkVelocityDeficit(BaseModel):
    """
    Model based on the TurbOPark model. For model details see
    https://github.com/OrstedRD/TurbOPark,
    https://github.com/OrstedRD/TurbOPark/blob/main/TurbOPark%20description.pdf, and
    Nygaard, Nicolai Gayle, et al. "Modelling cluster wakes and wind farm blockage."
    Journal of Physics: Conference Series. Vol. 1618. No. 6. IOP Publishing, 2020.
    """
    A: float = field(default=0.04)
    sigma_max_rel: float = field(default=4.0)
    overlap_gauss_interp: RegularGridInterpolator = field(init=False)

    def __attrs_post_init__(self) -> None:
        lookup_table_matlab_file = Path(__file__).parent / "turbopark_lookup_table.mat"
        lookup_table_file = scipy.io.loadmat(lookup_table_matlab_file)
        dist = lookup_table_file['overlap_lookup_table'][0][0][0][0]
        radius_down = lookup_table_file['overlap_lookup_table'][0][0][1][0]
        overlap_gauss = lookup_table_file['overlap_lookup_table'][0][0][2]
        self.overlap_gauss_interp = RegularGridInterpolator(
            (dist, radius_down),
            overlap_gauss,
            method='linear',
            bounds_error=False
        )

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        ambient_turbulence_intensity: np.ndarray,
        Cts: np.ndarray,
        rotor_diameter_i: np.ndarray,
        rotor_diameters: np.ndarray,
        i: int,
        deflection_field: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
    ) -> None:
        delta_total = np.zeros_like(u_initial)

        # Normalized distances along x between the turbine i and all other turbines
        # The downstream_mask is used to avoid negative numbers in the sqrt and the
        # subsequent runtime warnings.
        # Here self.NUM_EPS is to avoid precision issues with masking, and is slightly
        # larger than 0.0
        downstream_mask = np.array(x_i - x >= self.NUM_EPS)
        x_dist = (x_i - x) * downstream_mask / rotor_diameters

        # Radial distance between turbine i and the centerlines of wakes from all
        # real/image turbines
        r_dist = np.sqrt((y_i - (y + deflection_field)) ** 2 + (z_i - z) ** 2)
        r_dist_image = np.sqrt((y_i - (y + deflection_field)) ** 2 + (z_i - (-z)) ** 2)

        Cts[:,:,i:,:,:] = 0.00001

        # Characteristic wake widths from all turbines relative to turbine i
        dw = characteristic_wake_width(x_dist, ambient_turbulence_intensity, Cts, self.A)
        epsilon = 0.25 * np.sqrt(
            np.min( 0.5 * (1 + np.sqrt(1 - Cts)) / np.sqrt(1 - Cts), 3, keepdims=True )
        )
        sigma = rotor_diameters * (epsilon + dw)

        # Peak wake deficits
        val = 1 - Cts / (8 * (sigma / rotor_diameters) ** 2)
        C = 1 - np.sqrt(val)

        # Compute deficit for all turbines and mask to keep upstream and overlapping turbines
        effective_width = self.sigma_max_rel * sigma
        is_overlapping = effective_width / 2 + rotor_diameter_i / 2 > r_dist

        wtg_overlapping = np.array(x_dist > 0) * is_overlapping

        delta_real = np.empty(np.shape(u_initial)) * np.nan
        delta_image = np.empty(np.shape(u_initial)) * np.nan

        # Compute deficits for real turbines and for mirrored (image) turbines
        delta_real = C * wtg_overlapping * self.overlap_gauss_interp(
            (r_dist / sigma, rotor_diameter_i / 2 / sigma)
        )
        delta_image = C * wtg_overlapping * self.overlap_gauss_interp(
            (r_dist_image / sigma, rotor_diameter_i / 2 / sigma)
        )
        delta = np.concatenate((delta_real, delta_image), axis=2)

        delta_total[:, :, i, :, :] = np.sqrt(np.sum(np.nan_to_num(delta)**2, axis=2))

        return delta_total


def precalculate_overlap():
    # TODO: first implementation to generate wake overlap lookup table
    # (currently supplied by turbopark_lookup_table.mat.)
    # However, the result of this function doesn't generate the same
    # interpolant as the .mat file, so if used, needs to be corrected.
    dist = np.arange(0, 10, 1.0)
    radius_down = np.arange(0, 20, 1.0)
    overlap_gauss = np.zeros((len(dist), len(radius_down)))

    for i in range(len(dist)):
        for j in range(len(radius_down)):
            if radius_down[j] > 0:
                def fun(r, theta):
                    return r * np.exp(
                        -1 * (r ** 2 + dist[i] ** 2 - 2 * dist[i] * r * np.cos(theta)) / 2
                    )
                out = integrate.dblquad(fun, 0, radius_down[j], lambda x: 0, lambda x: 2 * np.pi)[0]
                out = out / (np.pi * radius_down[j] ** 2)
            else:
                out = np.exp(-(dist[i] ** 2) / 2)
            overlap_gauss[i, j] = out

    return dist, radius_down, overlap_gauss


def characteristic_wake_width(x_dist, TI, Cts, A):
    # Parameter values taken from S. T. Frandsen, “Risø-R-1188(EN) Turbulence
    # and turbulence generated structural loading in wind turbine clusters”
    # Risø, Roskilde, Denmark, 2007.
    c1 = 1.5
    c2 = 0.8

    alpha = TI * c1
    beta = c2 * TI / np.sqrt(Cts)

    dw = A * TI / beta * (
        np.sqrt((alpha + beta * x_dist) ** 2 + 1)
        - np.sqrt(1 + alpha ** 2)
        - np.log(
            ((np.sqrt((alpha + beta * x_dist) ** 2 + 1) + 1) * alpha)
            / ((np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x_dist))
        )
    )

    return dw
