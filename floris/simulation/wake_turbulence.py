# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np


class WakeTurbulence():
    """
    WakeTurbulence is the base class of the different wake velocity model
    classes.

    An instantiated WakeTurbulence object will import parameters used to
    calculate wake-added turbulence intensity from an upstream turbine,
    using one of several approaches.

    Returns:
        An instantiated WakeTurbulence object.
    """

    def __init__(self, ):
        self.requires_resolution = False
        self.model_string = None
        self.model_grid_resolution = None

    def __str__(self):
        return self.model_string


class Gauss(WakeTurbulence):
    """
    Gauss is a wake velocity subclass that contains objects related to the
    Gaussian wake model.

    Gauss is a subclass of
    :py:class:`floris.simulation.wake_velocity.WakeTurbulence` that is
    used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity. The Gaussian wake model includes a
    Gaussian wake velocity deficit profile in the y and z directions
    and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more
    information about the Gauss wake model theory, see:

    Abkar, M. and Porte-Agel, F. "Influence of atmospheric stability on
    wind-turbine wakes: A large-eddy simulation study." *Physics of
    Fluids*, 2015.

    Bastankhah, M. and Porte-Agel, F. "A new analytical model for
    wind-turbine wakes." *Renewable Energy*, 2014.

    Bastankhah, M. and Porte-Agel, F. "Experimental and theoretical
    study of wind turbine wakes in yawed conditions." *J. Fluid
    Mechanics*, 2016.

    Niayifar, A. and Porte-Agel, F. "Analytical modeling of wind farms:
    A new approach for power prediction." *Energies*, 2016.

    Dilip, D. and Porte-Agel, F. "Wind turbine wake mitigation through
    blade pitch offset." *Energies*, 2017.

    Args:
        parameter_dictionary: A dictionary as generated from the
            input_reader; it should have the following key-value pairs:

            -   **gauss**: A dictionary containing the following
                key-value pairs:

                -   **initial**: A float that is the initial ambient
                    turbulence intensity, expressed as a decimal
                    fraction.
                -   **constant**: A float that is the constant used to
                    scale the wake-added turbulence intensity.
                -   **ai**: A float that is the axial induction factor
                    exponent used in in the calculation of wake-added
                    turbulence.
                -   **downstream**: A float that is the exponent
                    applied to the distance downtream of an upstream
                    turbine normalized by the rotor diameter used in
                    the calculation of wake-added turbulence.


    Returns:
        An instantiated Gauss(WakeTurbulence) object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__()
        self.model_string = "gauss"
        model_dictionary = parameter_dictionary[self.model_string]

        # turbulence parameters
        self.ti_initial = float(model_dictionary["initial"])
        self.ti_constant = float(model_dictionary["constant"])
        self.ti_ai = float(model_dictionary["ai"])
        self.ti_downstream = float(model_dictionary["downstream"])

    def function(self, turb_u_wake, sorted_map, x_locations, y_locations,
                 z_locations, turbine, turbine_coord, flow_field):
        """
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified
        turbine, relative to the freestream velocities at the grid of
        points comprising the wind farm flow field.

        Args:
            turb_u_wake (np.array): u-component of turbine wake field
            sorted_map (list): sorted turbine_map (coord, turbine)
            x_locations: An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations: An array of floats that contains the grid
                coordinates of the flow field domain in the direction
                normal to x and parallel to the ground (m).
            z_locations: An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine: A :py:obj:`floris.simulation.turbine` object that
                represents the turbine creating the wake.
            turbine_coord: A :py:obj:`floris.utilities.Vec3` object
                containing the coordinate of the turbine creating the
                wake (m).
            flow_field: A :py:class:`floris.simulation.flow_field`
                object containing the flow field information for the
                wind farm.
        """

        # compute area overlap of wake on other turbines and update downstream
        # turbine turbulence intensities
        for coord_ti, turbine_ti in sorted_map:

            if coord_ti.x1 > turbine_coord.x1 and np.abs(
                    turbine_coord.x2 -
                    coord_ti.x2) < 2 * turbine.rotor_diameter:
                # only assess the effects of the current wake

                freestream_velocities = turbine_ti.calculate_swept_area_velocities(
                    flow_field.wind_direction, flow_field.u_initial, coord_ti,
                    x_locations, y_locations, z_locations)

                wake_velocities = turbine_ti.calculate_swept_area_velocities(
                    flow_field.wind_direction,
                    flow_field.u_initial - turb_u_wake, coord_ti, x_locations,
                    y_locations, z_locations)

                area_overlap = flow_field._calculate_area_overlap(
                    wake_velocities, freestream_velocities, turbine)
                if area_overlap > 0.0:
                    ti_initial = flow_field.turbulence_intensity

                    # turbulence intensity calculation based on Crespo et. al.
                    ti_calculation = self.ti_constant \
                        * turbine.aI**self.ti_ai \
                        * ti_initial**self.ti_initial \
                        * ((coord_ti.x1 - turbine_coord.x1) / turbine_ti.rotor_diameter)**self.ti_downstream

                    # Update turbulence intensity of downstream turbines
                    turbine_ti.turbulence_intensity = np.sqrt(
                        ti_calculation**2 + flow_field.turbulence_intensity**2)


class Ishihara(WakeTurbulence):
    """
    Ishihara is a wake velocity subclass that contains objects related to the
    Gaussian wake model that include a near-wake correction.

    Ishihara is a subclass of
    :py:class:`floris.simulation.wake_velocity.WakeTurbulence` that is
    used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity and a near wake correction. The Ishihara
    wake model includes a Gaussian wake velocity deficit profile in the y and z
    directions and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more info,
    see:

    Ishihara, Takeshi, and Guo-Wei Qian. "A new Gaussian-based analytical wake
    model for wind turbines considering ambient turbulence intensities and
    thrust coefficient effects." Journal of Wind Engineering and Industrial
    Aerodynamics 177 (2018): 275-292.

    Args:
        parameter_dictionary: A dictionary as generated from the
            input_reader; it should have the following key-value pairs:
            -   **ishihara**: A dictionary containing the following
                key-value pairs:

                -   **kstar**: A float that is a parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **epsilon**: A float that is a second parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **d**: constant coefficient used in calculation of              wake-added turbulence.
                -   **e**: linear coefficient used in calculation of                wake-added turbulence.
                -   **f**: near-wake coefficient used in calculation of             wake-added turbulence.

    Returns:
        An instantiated Ishihara(WakeTurbulence) object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__()
        self.model_string = "ishihara"
        model_dictionary = parameter_dictionary[self.model_string]

        # wake model parameter
        self.kstar = model_dictionary["kstar"]
        self.epsilon = model_dictionary["epsilon"]
        self.d = model_dictionary["d"]
        self.e = model_dictionary["e"]
        self.f = model_dictionary["f"]

    def function(self, turb_u_wake, sorted_map, x_locations, y_locations,
                 z_locations, turbine, turbine_coord, flow_field):
        """
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified
        turbine, relative to the freestream velocities at the grid of
        points comprising the wind farm flow field.

        Args:
            turb_u_wake (np.array): not used for the current turbulence model,      included for consistency of function form
            sorted_map (list): sorted turbine_map (coord, turbine)
            x_locations: An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations: An array of floats that contains the grid
                coordinates of the flow field domain in the direction
                normal to x and parallel to the ground (m).
            z_locations: An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine: A :py:obj:`floris.simulation.turbine` object that
                represents the turbine creating the wake (i.e. the 
                upstream turbine).
            turbine_coord: A :py:obj:`floris.utilities.Vec3` object
                containing the coordinate of the turbine creating the
                wake (m).
            deflection_field: An array of floats that contains the
                amount of wake deflection in meters in the y direction
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field`
                object containing the flow field information for the
                wind farm.
        """
        # compute area overlap of wake on other turbines and update downstream
        # turbine turbulence intensities
        for coord_ti, turbine_ti in sorted_map:

            if coord_ti.x1 > turbine_coord.x1 and np.abs(
                    turbine_coord.x2 -
                    coord_ti.x2) < 2 * turbine.rotor_diameter:
                # only assess the effects of the current wake

                # added turbulence model
                ti_initial = turbine.turbulence_intensity

                # turbine parameters
                D = turbine.rotor_diameter
                HH = turbine.hub_height
                Ct = turbine.Ct

                local_x = x_locations - turbine_coord.x1
                local_y = y_locations - turbine_coord.x2
                local_z = z_locations - turbine_coord.x3
                # coordinate info
                r = np.sqrt(local_y**2 + (local_z)**2)

                def parameter_value_from_dict(pdict, Ct, ti_initial):
                    return pdict['const'] * Ct**(pdict['Ct']) * ti_initial**(
                        pdict['TI'])

                kstar = parameter_value_from_dict(self.kstar, Ct, ti_initial)
                epsilon = parameter_value_from_dict(self.epsilon, Ct,
                                                    ti_initial)

                d = parameter_value_from_dict(self.d, Ct, ti_initial)
                e = parameter_value_from_dict(self.e, Ct, ti_initial)
                f = parameter_value_from_dict(self.f, Ct, ti_initial)

                k1 = np.cos(np.pi / 2 * (r / D - 0.5))**2
                k1[r / D > 0.5] = 1.0

                k2 = np.cos(np.pi / 2 * (r / D + 0.5))**2
                k2[r / D > 0.5] = 0.0

                # Representative wake width = \sigma / D
                wake_width = kstar * (local_x / D) + epsilon

                # Added turbulence intensity = \Delta I_1 (x,y,z)
                delta = ti_initial * np.sin(np.pi * (HH - local_z) / HH)**2
                delta[local_z >= HH] = 0.0
                ti_calculation = 1 / (
                    d + e * (local_x / D) + f * (1 + (local_x / D))**(-2)) * (
                        (k1 * np.exp(-(r - D / 2)**2 /
                                     (2 * (wake_width * D)**2))) +
                        (k2 * np.exp(-(r + D / 2)**2 /
                                     (2 * (wake_width * D)**2)))) - delta

                # Update turbulence intensity of downstream turbines
                turbine_ti.turbulence_intensity = np.sqrt(
                    ti_calculation**2 + flow_field.turbulence_intensity**2)
