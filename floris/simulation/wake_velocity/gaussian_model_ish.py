# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import cosd, sind, tand
import numpy as np


class GaussianModel():
    """
    This is the first draft of what will hopefully become the new gaussian class 
    Currently it contains a direct port of the Bastankhah gaussian class from previous
    A direct implementation of the Blondel model
    And a new GM model where we merge features a bit more of the two to ensure consistency with previous far-wake results
    of the Gaussian model, while implementing the Blondel model's smooth near-wake

    TODO: This needs to be much more expanded and including full references

    [1] Abkar, M. and Porte-Agel, F. "Influence of atmospheric stability on
    wind-turbine wakes: A large-eddy simulation study." *Physics of
    Fluids*, 2015.

    [2] Bastankhah, M. and Porte-Agel, F. "A new analytical model for
    wind-turbine wakes." *Renewable Energy*, 2014.

    [3] Bastankhah, M. and Porte-Agel, F. "Experimental and theoretical
    study of wind turbine wakes in yawed conditions." *J. Fluid
    Mechanics*, 2016.

    [4] Niayifar, A. and Porte-Agel, F. "Analytical modeling of wind farms:
    A new approach for power prediction." *Energies*, 2016.

    [5] Dilip, D. and Porte-Agel, F. "Wind turbine wake mitigation through
    blade pitch offset." *Energies*, 2017.

    [6] Blondel, F. and Cathelain, M. "An alternative form of the
    super-Gaussian wind turbine wake model." *Wind Energy Science Disucssions*,
    2020.
    Notes to be written (merged)
    """    

    @staticmethod
    def mask_upstream_wake(y_locations, turbine_coord, yaw):
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1
        return xR, yR

    @staticmethod
    def initial_velocity_deficits(U_local, Ct):
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - Ct)))
        u0 = U_local * np.sqrt(1 - Ct)
        return uR, u0

    @staticmethod
    def initial_wake_expansion(turbine, U_local, veer, uR, u0):
        yaw = -1 * turbine.yaw_angle 
        sigma_z0 = turbine.rotor_diameter * 0.5 * np.sqrt( uR / (U_local + u0) )
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)
        return sigma_y0, sigma_z0

    @staticmethod
    def gaussian_function(U, C, r, n, sigma):
        return U * C * np.exp( -1 * r**n / (2 * sigma**2) )
