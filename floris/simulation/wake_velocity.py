# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from ..utilities import Vec3
from ..utilities import cosd, sind, tand
import copy


class WakeVelocity():
    """
    WakeVelocity is the base class of the different wake velocity model 
    classes.

    An instantiated WakeVelocity object will import parameters used to 
    calculate wake-added turbulence intensity from an upstream turbine, 
    using the approach of Crespo, A. and Herna, J., "Turbulence 
    characteristics in wind-turbine wakes." *J. Wind Eng Ind Aerodyn*. 
    1996.

    Args:
        parameter_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs:

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
        An instantiated WakeVelocity object.
    """

    def __init__(self, parameter_dictionary):
        self.requires_resolution = False
        self.model_string = None
        self.model_grid_resolution = None
        self.parameter_dictionary = parameter_dictionary

        # turbulence parameters
        turbulence_intensity = self.parameter_dictionary["turbulence_intensity"]
        self.ti_initial = float(turbulence_intensity["initial"])
        self.ti_constant = float(turbulence_intensity["constant"])
        self.ti_ai = float(turbulence_intensity["ai"])
        self.ti_downstream = float(turbulence_intensity["downstream"])
    
    def _get_model_dict(self):
        if self.model_string not in self.parameter_dictionary.keys():
            raise KeyError("The {} wake model was".format(self.model_string) +
                " instantiated but the model parameters were not found in the" +
                " input file or dictionary under" +
                " 'wake.properties.parameters.{}'.".format(self.model_string))
        return self.parameter_dictionary[self.model_string]

class Jensen(WakeVelocity):
    """
    Wake velocity deficit model based on the Jensen model.

    Jensen is a subclass of :py:class:`floris.simulation.wake_velocity.WakeVelocity` that is 
    used to compute the wake velocity deficit based on the classic 
    Jensen/Park model. See Jensen, N. O., "A note on wind generator 
    interaction." Tech. Rep. Risø-M-2411, Risø, 1983.

    Args:
        parameter_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs used to calculate wake-added 
                turbulence intensity from an upstream turbine, using 
                the approach of Crespo, A. and Herna, J. "Turbulence 
                characteristics in wind-turbine wakes." *J. Wind Eng 
                Ind Aerodyn*. 1996.:

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

            -   **jensen**: A dictionary containing the following 
                key-value pairs:

                -   **we**: A float that is the linear wake decay 
                    constant that defines the cone boundary for the 
                    wake as well as the velocity deficit. D/2 +/- we*x 
                    is the cone boundary for the wake.

    Returns:
        An instantiated Jensen object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "jensen"
        model_dictionary = self._get_model_dict()
        self.we = float(model_dictionary["we"])

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        """
        Using the Jensen wake model, this method calculates and returns 
        the wake velocity deficits, caused by the specified turbine, 
        relative to the freestream velocities at the grid of points 
        comprising the wind farm flow field.

        Args:
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
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        # define the boundary of the wake model ... y = mx + b
        m = self.we
        x = x_locations - turbine_coord.x1
        b = turbine.rotor_radius

        boundary_line = m * x + b

        y_upper = boundary_line + turbine_coord.x2 + deflection_field
        y_lower = -1 * boundary_line + turbine_coord.x2 + deflection_field

        z_upper = boundary_line + turbine.hub_height
        z_lower = -1 * boundary_line + turbine.hub_height

        # calculate the wake velocity
        c = (turbine.rotor_diameter /
             (2 * self.we * (x_locations - turbine_coord.x1) + turbine.rotor_diameter))**2

        # filter points upstream and beyond the upper and lower bounds of the wake
        c[x_locations - turbine_coord.x1 < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0
        c[z_locations > z_upper] = 0
        c[z_locations < z_lower] = 0

        return 2 * turbine.aI * c * flow_field.u_initial, np.zeros(np.shape(flow_field.u_initial)), np.zeros(np.shape(flow_field.u_initial))


class MultiZone(WakeVelocity):
    """
    Floris is a subclass of 
    :py:class:`floris.simulation.wake_velocity.WakeVelocity` that is 
    used to compute the wake velocity deficit based on the original 
    multi-zone FLORIS model. See: 

    Gebraad, P. M. O. et al., "A Data-driven model for wind plant power 
    optimization by yaw control." *Proc. American Control Conference*, 
    Portland, OR, 2014.

    Gebraad, P. M. O. et al., "Wind plant power optimization through 
    yaw control using a parametric model for wake effects - a CFD 
    simulation study." *Wind Energy*, 2016.

    Args:
        parameter_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs used to calculate wake-added 
                turbulence intensity from an upstream - turbine, using 
                the approach of Crespo, A. and Herna, J. "Turbulence 
                characteristics in wind-turbine wakes." *J. Wind Eng 
                Ind Aerodyn*. 1996.:

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

            - **floris**: A dictionary containing the following 
                key-value pairs:

                -   **me**: A list of three floats that help determine 
                    the slope of the diameters of the three wake zones 
                    (near wake, far wake, mixing zone) as a function of 
                    downstream distance.
                -   **we**: A float that is the scaling parameter used 
                    to adjust the wake expansion, helping to determine 
                    the slope of the diameters of the three wake zones 
                    as a function of downstream distance, as well as 
                    the recovery of the velocity deficits in the wake 
                    as a function of downstream distance.
                -   **aU**: A float that is a parameter used to 
                    determine the dependence of the wake velocity 
                    deficit decay rate on the rotor yaw angle.
                -   **bU**: A float that is another parameter used to 
                    determine the dependence of the wake velocity 
                    deficit decay rate on the rotor yaw angle.
                -   **mU**: A list of three floats that are parameters 
                    used to determine the dependence of the wake 
                    velocity deficit decay rate for each of the three 
                    wake zones on the rotor yaw angle.

    Returns:
        An instantiated Floris object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "multizone"
        model_dictionary = self._get_model_dict()
        self.me = [float(n) for n in model_dictionary["me"]]
        self.we = float(model_dictionary["we"])
        self.aU = float(model_dictionary["aU"])
        self.bU = float(model_dictionary["bU"])
        self.mU = [float(n) for n in model_dictionary["mU"]]

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        """
        Using the original FLORIS multi-zone wake model, this method 
        calculates and returns the wake velocity deficits, caused by 
        the specified turbine, relative to the freestream velocities at 
        the grid of points comprising the wind farm flow field.

        Args:
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
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        mu = self.mU / cosd(self.aU + self.bU * turbine.yaw_angle)

        # distance from wake centerline
        rY = abs(y_locations - (turbine_coord.x2 + deflection_field))
        # rZ = abs(z_locations - (turbine.hub_height))
        dx = x_locations - turbine_coord.x1

        # wake zone diameters
        nearwake = turbine.rotor_radius + self.we * self.me[0] * dx
        farwake = turbine.rotor_radius + self.we * self.me[1] * dx
        mixing = turbine.rotor_radius + self.we * self.me[2] * dx

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= nearwake
        c += mask * (turbine.rotor_diameter /
                     (turbine.rotor_diameter + 2 * self.we * mu[0] * dx))**2
        #mask = rZ <= nearwake
        #c += mask * (radius / (radius + we * mu[0] * dx))**2

        # far wake zone
        # ^ is XOR, x^y:
        #   Each bit of the output is the same as the corresponding bit in x
        #   if that bit in y is 0, and it's the complement of the bit in x
        #   if that bit in y is 1.
        # The resulting mask is all the points in far wake zone that are not
        # in the near wake zone
        mask = (rY <= farwake) ^ (rY <= nearwake)
        c += mask * (turbine.rotor_diameter /
                     (turbine.rotor_diameter + 2 * self.we * mu[1] * dx))**2
        #mask = (rZ <= farwake) ^ (rZ <= nearwake)
        #c += mask * (radius / (radius + we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= mixing) ^ ((rY <= farwake) | (rY <= nearwake))
        c += mask * (turbine.rotor_diameter /
                     (turbine.rotor_diameter + 2 * self.we * mu[2] * dx))**2
        #mask = (rZ <= mixing) ^ ((rZ <= farwake) | (rZ <= nearwake))
        #c += mask * (radius / (radius + we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x1 < 0] = 0

        return 2 * turbine.aI * c * flow_field.wind_map.grid_wind_speed, np.zeros(np.shape(c)), np.zeros(np.shape(c))


class Gauss(WakeVelocity):
    """
    Gauss is a wake velocity subclass that contains objects related to the 
    Gaussian wake model.

    Gauss is a subclass of 
    :py:class:`floris.simulation.wake_velocity.WakeVelocity` that is 
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

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs used to calculate wake-added 
                turbulence intensity from an upstream turbine, using 
                the approach of Crespo, A. and Herna, J. "Turbulence 
                characteristics in wind-turbine wakes." *J. Wind Eng 
                Ind Aerodyn*. 1996.:

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

            -   **gauss**: A dictionary containing the following 
                key-value pairs:

                -   **ka**: A float that is a parameter used to 
                    determine the linear relationship between the 
                    turbulence intensity and the width of the Gaussian 
                    wake shape.
                -   **kb**: A float that is a second parameter used to 
                    determine the linear relationship between the 
                    turbulence intensity and the width of the Gaussian 
                    wake shape.
                -   **alpha**: A float that is a parameter that 
                    determines the dependence of the downstream 
                    boundary between the near wake and far wake region 
                    on the turbulence intensity.
                -   **beta**: A float that is a parameter that 
                    determines the dependence of the downstream 
                    boundary between the near wake and far wake region 
                    on the turbine's induction factor.

    Returns:
        An instantiated Gauss object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "gauss"
        model_dictionary = self._get_model_dict()

        # wake expansion parameters
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])

        # near wake parameters
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        """
        Using the Gaussian wake model, this method calculates and 
        returns the wake velocity deficits, caused by the specified 
        turbine, relative to the freestream velocities at the grid of 
        points comprising the wind farm flow field.

        Args:
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
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        # veer (degrees)
        veer = flow_field.wind_veer

        # added turbulence model
        TI = turbine.current_turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # wake deflection
        delta = deflection_field

        # initial velocity deficits
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - (Ct))))
        u0 = U_local * np.sqrt(1 - Ct)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        # quantity that determines when the far wake starts
        x0 = D * (cosd(yaw) * (1 + np.sqrt(1 - Ct))) / (np.sqrt(2) \
            * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) \
            + turbine_coord.x1

        # wake expansion parameters
        ky = self.ka * TI + self.kb
        kz = self.ka * TI + self.kb

        # compute velocity deficit
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # velocity deficit in the near wake
        sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_z0

        sigma_y[x_locations < xR] = 0.5 * D
        sigma_z[x_locations < xR] = 0.5 * D

        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        velDef = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef[x_locations < xR] = 0
        velDef[x_locations > x0] = 0

        # wake expansion in the lateral (y) and the vertical (z)
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0

        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        # compute velocities in the far wake
        velDef1 = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef1[x_locations < x0] = 0

        return np.sqrt(velDef**2 + velDef1**2), np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))


class Curl(WakeVelocity):
    """
    Curl is a wake velocity subclass that contains objects related to 
    the Curled Wake model.

    Curl is a subclass of 
    :py:class:`floris.simulation.wake_velocity.WakeVelocity` that is 
    used to compute the wake velocity deficit based on the curled wake 
    model developed by Martinez-Tossas et al. The curled wake model 
    includes the change in the shape of the wake profile under yawed 
    conditions due to vortices that are shed from the rotor plane of a 
    yawed turbine. The model includes the impact of turbulence 
    intensity, wind veer, and the tip-speed ratio of the turbine. For 
    more information about the curled wake model theory, see: 
    Martinez-Tossas, L. A. et al. "The aerodynamics of the curled wake: 
    a simplified model in view of flow control." *Wind Energy Science*, 
    2019. 

    For more information about the impact of the curled wake behavior 
    on wake steering, see: Fleming, P. et al. "A simulation study 
    demonstreating the importance of large-scale trailing vortices in 
    wake steering." *Wind Energy Science*, 2018. 

    Args:
        parameter_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs used to calculate wake-added 
                turbulence intensity from an upstream turbine, using 
                the approach of Crespo, A. and Herna, J. "Turbulence 
                characteristics in wind-turbine wakes." *J. Wind Eng 
                Ind Aerodyn*. 1996.:

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

            -   **curl**: A dictionary containing the following 
                key-value pairs:

                -   **model_grid_resolution**: A list of three floats 
                    that define the flow field grid resolution in the x,
                    y, and z directions used for the curl wake model 
                    calculations. The grid resolution is specified as 
                    the number of grid points in the flow field domain 
                    in the x, y, and z directions. 
                -   **initial_deficit**: A float that, along with the 
                    freestream velocity and the turbine's induction 
                    factor, is used to determine the initial wake 
                    velocity deficit immediately downstream of the 
                    rotor.
                -   **dissipation**: A float that is a scaling 
                    parameter that determines the amount of dissipation 
                    of the vortices with downstream distance.
                -   **veer_linear**: A float that describes the amount 
                    of linear wind veer. This parameter defines the 
                    linear change in the V velocity between the ground 
                    and hub height, and therefore determines the slope 
                    of the change in the V velocity with height. 

    Returns:
        An instantiated Curl object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "curl"
        model_dictionary = self._get_model_dict()
        self.model_grid_resolution = Vec3(
            model_dictionary["model_grid_resolution"])
        self.initial_deficit = float(model_dictionary["initial_deficit"])
        self.dissipation = float(model_dictionary["dissipation"])
        self.veer_linear = float(model_dictionary["veer_linear"])
        self.requires_resolution = True

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        """
        Using the Curl wake model, this method calculates and returns 
        the wake velocity deficits, caused by the specified turbine, 
        relative to the freestream velocities at the grid of points 
        comprising the wind farm flow field.

        Args:
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
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        # parameters available for tuning to match high-fidelity data
        # parameter for defining initial velocity deficity in the flow field at a turbine
        intial_deficit = self.initial_deficit
        # scaling parameter that adjusts the amount of dissipation of the vortexes
        dissipation = self.dissipation
        # parameter that defines the wind velocity of veer at 0 meters height
        veer_linear = self.veer_linear

        # setup x and y grid information
        x = np.linspace(np.min(x_locations), np.max(
            x_locations), int(self.model_grid_resolution.x1))
        y = np.linspace(np.min(y_locations), np.max(
            y_locations), int(self.model_grid_resolution.x2))

        # find the x-grid location closest to the current turbine
        idx = np.min(np.where(x >= turbine_coord.x1))

        # initialize the flow field
        uw = np.zeros(
            (
                int(self.model_grid_resolution.x1),
                int(self.model_grid_resolution.x2),
                int(self.model_grid_resolution.x3)
            )
        )

        # determine values to create a rotor mask for velocities
        y1 = y_locations[idx, :, :] - turbine_coord.x2
        z1 = z_locations[idx, :, :] - turbine.hub_height
        r1 = np.sqrt(y1**2 + z1**2)

        # add initial velocity deficit at the rotor to the flow field
        uw_initial = -1 * (flow_field.wind_map.grid_wind_speed * intial_deficit * turbine.aI)
        uw[idx, :, :] = gaussian_filter(
            uw_initial[idx,:,:] * (r1 <= turbine.rotor_diameter / 2), sigma=1)

        # enforce the boundary conditions
        uw[idx,  0, :] = 0.0
        uw[idx, :,  0] = 0.0
        uw[idx, -1, :] = 0.0
        uw[idx, :, -1] = 0.0

        # TODO: explain?
        uw = -1 * uw

        # parameters to simplify the code
        # diameter of the turbine rotor from the input file
        D = turbine.rotor_diameter
        Ct = turbine.Ct                                 # thrust coefficient of the turbine
        yaw = turbine.yaw_angle                         # yaw angle of the turbine
        HH = turbine.hub_height                         # hub height of the turbine
        # the free-stream velocity of the flow field
        Uinf = flow_field.wind_map.grid_wind_speed[idx,:,:]
        # the tip-speed ratior of the turbine
        TSR = turbine.tsr
        # the axial induction factor of the turbine
        aI = turbine.aI
        # initial velocities in the stream-wise, span-wise, and vertical direction
        U, V, W = flow_field.u, flow_field.v, flow_field.w
        # the tilt angle of the rotor of the turbine
        tilt = turbine.tilt_angle

        # calculate the curled wake effects due to the yaw and tilt of the turbine
        Gamma_Yaw = flow_field.air_density * np.pi * D / 8 * Ct * \
            turbine.average_velocity * sind(yaw)
        if turbine.yaw_angle != 0.0:
            YawFlag = 1
        else:
            YawFlag = 0
        Gamma_Tilt = flow_field.air_density * np.pi * D / 8 * Ct * \
            turbine.average_velocity * sind(tilt)
        if turbine.tilt_angle != 0.0:
            TiltFlag = 1
        else:
            TiltFlag = 0

        Gamma = Gamma_Yaw + Gamma_Tilt

        # calculate the curled wake effects due to the rotation of the turbine rotor
        Gamma_wake_rotation = 2 * np.pi * D * (aI - aI**2) * Uinf / TSR

        # =======================================================================
        # add curl Elliptic
        # =======================================================================
        eps = 0.2 * D

        # distribute rotation across the blade
        z_vector = np.linspace(0, D / 2, 100)

        # length of each section dz
        dz = z_vector[1] - z_vector[0]

        # scale the circulation of each section dz
        if yaw != 0 or tilt != 0:
            Gamma0 = 4 / np.pi * Gamma
        else:
            Gamma0 = 0.0

        # loop through all the vortices from an elliptic wind distribution
        # skip the last point because it has zero circulation
        for z in z_vector[:-1]:

            # Compute the non-dimensional circulation
            Gamma = (-4 * Gamma0 * z * dz /
                     (D**2 * np.sqrt(1 - (2 * z / D)**2)))

            # locations of the tip vortices
            # top
            y_vortex_1 = turbine_coord.x2 + z * TiltFlag
            z_vortex_1 = HH + z * YawFlag

            # bottom
            y_vortex_2 = turbine_coord.x2 - z * TiltFlag
            z_vortex_2 = HH - z * YawFlag

            # vortex velocities
            # top
            v1, w1 = self._vortex(flow_field.y[idx, :, :] - y_vortex_1, flow_field.z[idx, :, :]
                                  - z_vortex_1, flow_field.x[idx, :, :] - turbine_coord.x1, -Gamma, eps, Uinf)
            # bottom
            v2, w2 = self._vortex(flow_field.y[idx, :, :] - y_vortex_2, flow_field.z[idx, :, :]
                                  - z_vortex_2, flow_field.x[idx, :, :] - turbine_coord.x1, Gamma, eps, Uinf)

            # add ground effects
            v3, w3 = self._vortex(flow_field.y[idx, :, :] - y_vortex_1, flow_field.z[idx, :, :]
                                  + z_vortex_1, flow_field.x[idx, :, :] - turbine_coord.x1, Gamma, eps, Uinf)
            v4, w4 = self._vortex(flow_field.y[idx, :, :] - y_vortex_2, flow_field.z[idx, :, :]
                                  + z_vortex_2, flow_field.x[idx, :, :] - turbine_coord.x1, -Gamma, eps, Uinf)

            V[idx, :, :] += v1 + v2 + v3 + v4
            W[idx, :, :] += w1 + w2 + w3 + w4

        # add wake rotation
        v5, w5 = self._vortex(flow_field.y[idx, :, :] - turbine_coord.x2, flow_field.z[idx, :, :]
                              - turbine.hub_height, flow_field.x[idx, :, :]
                              - turbine_coord.x1, Gamma_wake_rotation, 0.2 * D, Uinf) \
            * (np.sqrt((flow_field.y[idx, :, :] - turbine_coord.x2)**2
                       + (flow_field.z[idx, :, :] - turbine.hub_height)**2) <= D/2)
        v6, w6 = self._vortex(flow_field.y[idx, :, :] - turbine_coord.x2, flow_field.z[idx, :, :]
                              + turbine.hub_height, flow_field.x[idx, :, :]
                              - turbine_coord.x1, -Gamma_wake_rotation, 0.2 * D, Uinf) \
            * (np.sqrt((flow_field.y[idx, :, :] - turbine_coord.x2)**2
                       + (flow_field.z[idx, :, :] - turbine.hub_height)**2) <= D/2)
        V[idx, :, :] += v5 + v6
        W[idx, :, :] += w5 + w6

        # decay the vortices as they move downstream
        lmda = 15
        kappa = 0.41
        lm = kappa * z / (1 + kappa * z / lmda)
        dudz_initial = np.gradient(U, z, axis=2)
        nu = lm**2 * np.abs(dudz_initial[0, :, :])

        for i in range(idx, len(x) - 1):
            V[i + 1, :, :] = V[idx, :, :] * eps**2 \
                / (4 * nu * (flow_field.x[i, :, :]
                             - turbine_coord.x1) / Uinf + eps**2)
            W[i + 1, :, :] = W[idx, :, :] * eps**2 \
                / (4 * nu * (flow_field.x[i, :, :]
                             - turbine_coord.x1) / Uinf + eps**2)

        # simple implementation of linear veer, added to the V component of the flow field
        z = np.linspace(
            np.min(z_locations),
            np.max(z_locations),
            int(self.model_grid_resolution.x3)
        )
        z_min = HH
        b_veer = veer_linear
        m_veer = -b_veer / z_min

        v_veer = m_veer * z + b_veer

        for i in range(len(z) - 1):
            V[:, :, i] = V[:, :, i] + v_veer[i]

        # ===========================================================================================
        # SOLVE CURL
        # ===========================================================================================
        dudz_initial = np.gradient(U, axis=2) \
            / np.gradient(z_locations, axis=2)

        ti_initial = flow_field.wind_map.grid_turbulence_intensity[idx,:,:]

        # turbulence intensity parameters stored in floris.json
        ti_i = self.ti_initial
        ti_constant = self.ti_constant
        ti_ai = self.ti_ai
        ti_downstream = self.ti_downstream

        for i in range(idx + 1, len(x)):

            # compute the change in x
            dx = x[i] - x[i - 1]

            dudy = np.gradient(uw[i - 1, :, :], axis=0) \
                / np.gradient(y_locations[i - 1, :, :], axis=0)
            dudz = np.gradient(uw[i - 1, :, :], axis=1) \
                / np.gradient(z_locations[i - 1, :, :], axis=1)

            gradU = np.gradient(np.gradient(uw[i - 1, :, :], axis=0), axis=0) \
                / np.gradient(y_locations[i - 1, :, :], axis=0)**2 \
                + np.gradient(np.gradient(uw[i - 1, :, :], axis=1), axis=1) \
                / np.gradient(z_locations[i - 1, :, :], axis=1)**2

            lm = kappa * z / (1 + kappa * z / lmda)
            nu = lm**2 * np.abs(dudz_initial[i - 1, :, :])

            # turbulence intensity calculation based on Crespo et. al.
            ti_local = 10*ti_constant \
                * turbine.aI**ti_ai \
                * ti_initial**ti_i \
                * ((x[i] - turbine_coord.x1) / turbine.rotor_diameter)**ti_downstream

            # solve the marching problem for u, v, and w
            uw[i, :, :] = uw[i - 1, :, :] + (dx / (U[i - 1, :, :])) \
                * (-V[i - 1, :, :] * dudy - W[i - 1, :, :] * dudz
                   + dissipation * D * nu * ti_local * gradU)
            # enforce boundary conditions
            uw[i, :, 0] = np.zeros(len(y))
            uw[i, 0, :] = np.zeros(len(z))

        uw[x_locations < turbine_coord.x1] = 0.0

        return uw, V, W

    def _vortex(self, x, y, z, Gamma, eps, U):
        # compute the vortex velocity
        v = (Gamma / (2 * np.pi)) * (y / (x**2 + y**2)) \
            * (1 - np.exp(-(x**2 + y**2) / eps**2))
        w = -(Gamma / (2 * np.pi)) * (x / (x**2 + y**2)) \
            * (1 - np.exp(-(x**2 + y**2) / eps**2))

        return v, w


class GaussCurlHybrid(WakeVelocity):

    def __init__(self, parameter_dictionary):
        # TODO: update docstring

        super().__init__(parameter_dictionary)
        self.model_string = "gauss_curl_hybrid"
        model_dictionary = self._get_model_dict()

        # wake expansion parameter
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])

        # near wake parameter
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])

        if 'use_yar' in model_dictionary:
            self.use_yar = bool(model_dictionary["use_yar"])
        else:
            # TODO: introduce logging
            print('Using default option of not applying added yaw-added recovery (use_yar=False)')
            self.use_yar = False

        if 'yaw_rec_alpha' in model_dictionary:
            self.yaw_rec_alpha = bool(model_dictionary["yaw_rec_alpha"])
        else:
            self.yaw_rec_alpha = 0.03
            # TODO: introduce logging
            print('Using default option yaw_rec_alpha: %.2f' % self.yaw_rec_alpha)

        if 'eps_gain' in model_dictionary:
            self.eps_gain = bool(model_dictionary["eps_gain"])
        else:
            self.eps_gain = 0.3 # SOWFA SETTING (note this will be multiplied by D in function)
            # TODO: introduce logging
            print('Using default option eps_gain: %.1f' % self.eps_gain)

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        """
        Using the Gauss-Curl hybrid wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified turbine, 
        relative to the freestream velocities at the grid of points 
        comprising the wind farm flow field.

        Args:
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
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        # veer (degrees)
        veer = flow_field.wind_veer

        # added turbulence model
        TI = turbine.current_turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # wake deflection
        delta = deflection_field

        # initial velocity deficits
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - (Ct))))
        u0 = U_local * np.sqrt(1 - Ct)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        # quantity that determines when the far wake starts
        x0 = D * (cosd(yaw) * (1 + np.sqrt(1 - Ct))) / (np.sqrt(2) \
            * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) \
            + turbine_coord.x1

        # wake expansion parameters
        ky = self.ka * TI + self.kb
        kz = self.ka * TI + self.kb

        # compute velocity deficit
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # velocity deficit in the near wake
        sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_z0

        sigma_y[x_locations < xR] = 0.5 * D
        sigma_z[x_locations < xR] = 0.5 * D

        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        velDef = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef[x_locations < xR] = 0
        velDef[x_locations > x0] = 0

        # wake expansion in the lateral (y) and the vertical (z)
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0

        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        # compute velocities in the far wake
        velDef1 = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef1[x_locations < x0] = 0

        U = np.sqrt(velDef**2 + velDef1**2)

        # compute the spanwise and vertical velocity components
        V, W = self._velocity_components(turbine_coord, turbine, flow_field, x_locations, y_locations, z_locations)

        # If indicated, include the added yaw recovery option
        if self.use_yar:

            # compute the velocity without modification
            U1 = U_local - U

            # set dimensions
            xLocs = (x_locations - turbine_coord.x1)
            yLocs = y_locations - turbine_coord.x2
            # zLocs = z_locations
            D = turbine.rotor_diameter

            numerator = - (W * xLocs * np.abs(yLocs))
            denom = np.pi * ((self.yaw_rec_alpha * xLocs + D/2) ** 2)
            U2 = numerator/denom

            # add velocity modification from yaw (U2)
            U_total = U1 + np.nan_to_num(U2)

            # turn it back into a deficit
            U = U_local - U_total

            # zero out anything before the turbine
            U[x_locations < turbine_coord.x1] = 0

        return U, V, W

    def _velocity_components(self, coord, turbine, flow_field, x_locations, y_locations, z_locations):

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = turbine.yaw_angle
        Ct = turbine.Ct
        TSR = turbine.tsr
        aI = turbine.aI

        # flow parameters
        rho = flow_field.air_density

        # Update to wind map
        # Uinf = flow_field.wind_speed
        Uinf = np.mean(flow_field.wind_map.input_speed) # TODO Is this right?

        # top point of the rotor
        dist_top = np.sqrt((coord.x1 - x_locations) ** 2 + ((coord.x2) - y_locations) ** 2 + (
                    z_locations - (turbine.hub_height + D / 2)) ** 2)
        idx_top = np.where(dist_top == np.min(dist_top))

        # bottom point of the rotor
        dist_bottom = np.sqrt((coord.x1 - x_locations) ** 2 + ((coord.x2) - y_locations) ** 2 + (
                z_locations - (turbine.hub_height - D / 2)) ** 2)
        idx_bottom = np.where(dist_bottom == np.min(dist_bottom))

        if len(idx_top) > 1:
            idx_top = idx_top[0]
        if len(idx_bottom) > 1:
            idx_bottom = idx_bottom[0]

        scale = 1.0
        Gamma_top = scale * (np.pi / 8) * rho * D * turbine.average_velocity * Ct * sind(yaw) * cosd(yaw) ** 2
        Gamma_bottom = scale*(np.pi/8) * rho * D * turbine.average_velocity * Ct * sind(yaw) * cosd(yaw)**2
        Gamma_wake_rotation = 0.5 * 2 * np.pi * D * (aI - aI ** 2) * turbine.average_velocity / TSR

        # compute the spanwise and vertical velocities induced by yaw
        # Use set value
        eps = self.eps_gain * D

        # decay the vortices as they move downstream - using mixing length
        lmda = D/8 #D/4 #D/4 #D/2
        kappa = 0.41
        lm = kappa * z_locations / (1 + kappa * z_locations / lmda)
        z = np.linspace(np.min(z_locations),np.max(z_locations),np.shape(flow_field.u_initial)[2])
        dudz_initial = np.gradient(flow_field.u_initial, z, axis=2)
        nu = lm ** 2 * np.abs(dudz_initial[0, :, :])

        # top vortex
        yLocs = y_locations+0.01 - (coord.x2)
        zLocs = z_locations+0.01 - (HH + D/2)
        V1 = (((yLocs * Gamma_top) / (2 * np.pi * (yLocs**2 + zLocs**2))) * (1 - np.exp(-(yLocs**2 + zLocs**2)/(eps**2))) ) * \
            eps**2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps**2)

        W1 = ((zLocs * Gamma_top) / (2 * np.pi * (yLocs**2 + zLocs**2))) * (1 - np.exp(-(yLocs**2 + zLocs**2)/(eps**2))) * \
            eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # bottom vortex
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 - (HH - D/2)
        V2 = (((yLocs * -Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                    1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))) * \
            eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W2 = ((zLocs * -Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                    1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) * \
            eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # top vortex - ground
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 + (HH + D/2)
        V3 = (((yLocs * -Gamma_top) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                    1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) * \
             eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W3 = ((zLocs * -Gamma_top) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                    1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) * \
             eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # bottom vortex - ground
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 + (HH - D / 2)
        V4 = (((yLocs * Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) * \
             eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W4 = ((zLocs * Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) * \
             eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # wake rotation vortex
        yLocs = y_locations + 0.01 - coord.x2
        zLocs = z_locations + 0.01 - HH
        V5 = (((yLocs * Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                    1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) * \
            eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W5 = ((zLocs * Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                    1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) * \
            eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # wake rotation vortex - ground effect
        yLocs = y_locations + 0.01 - coord.x2
        zLocs = z_locations + 0.01 + HH
        V6 = (((yLocs * Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) * \
             eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W6 = ((zLocs * Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) * (
                1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) * \
             eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # total spanwise velocity
        V = V1 + V2 + V3 + V4 + V5 + V6

        # total vertical velocity
        W = W1 + W2 + W3 + W4 + W5 + W6

        # compute velocity deficit
        # yR = y_locations - coord.x2
        # xR = yR * tand(yaw) + coord.x1
        V[x_locations < coord.x1+10] = 0.0
        W[x_locations < coord.x1+10] = 0.0

        # cut off in the spanwise direction
        V[np.abs(y_locations-coord.x2) > D] = 0.0
        W[np.abs(y_locations-coord.x2) > D] = 0.0

        return V, W
