# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
from ...utilities import cosd, sind, tand


class VelocityDeflection():
    """
    Base VelocityDeflection object class. Subclasses are:

    Each subclass has specific functional requirements. Refer to the
    each VelocityDeflection subclass for further detail.
    """
    def __init__(self, parameter_dictionary):
        self.model_string = None

        self.parameter_dictionary = parameter_dictionary

        if 'use_ss' in self.parameter_dictionary:
            self.use_ss = bool(self.parameter_dictionary["use_ss"])
        else:
            # TODO: introduce logging
            print(
                'Using default option of not applying gch-based secondary ' + 
                'steering (use_ss=False)'
            )
            self.use_ss = False

        if 'eps_gain' in self.parameter_dictionary:
            self.eps_gain = bool(self.parameter_dictionary["eps_gain"])
        else:
            # SOWFA SETTING (note this will be multiplied by D in function)
            self.eps_gain = 0.3  
            # TODO: introduce logging
            print('Using default option eps_gain: %.1f' % self.eps_gain)

    def _get_model_dict(self):
        if self.model_string not in self.parameter_dictionary.keys():
            raise KeyError("The {} wake model was".format(self.model_string) +
                " instantiated but the model parameters were not found in the" +
                " input file or dictionary under" +
                " 'wake.properties.parameters.{}'.".format(self.model_string))
        return self.parameter_dictionary[self.model_string]

    def calculate_effective_yaw_angle(self, x_locations, y_locations,
                                      z_locations, turbine, coord, flow_field):
        if self.use_ss:
            # turbine parameters
            Ct = turbine.Ct
            D = turbine.rotor_diameter
            HH = turbine.hub_height
            aI = turbine.aI
            TSR = turbine.tsr

            V = flow_field.v
            W = flow_field.w

            yLocs = y_locations - coord.x2
            zLocs = z_locations - (HH)

            # Use set value
            eps = self.eps_gain * D
            # TODO Is this right below?
            Uinf = np.mean(flow_field.wind_map.input_speed) 

            dist = np.sqrt(yLocs**2 + zLocs**2)
            xLocs = np.abs(x_locations - coord.x1)
            idx = np.where((dist < D / 2) & (xLocs < D / 4) \
                            & (np.abs(yLocs) > 0.1))

            Gamma = V[idx] * ((2 * np.pi) * (yLocs[idx]**2 + zLocs[idx]**2)) \
                / (yLocs[idx] * (1 - np.exp(-(yLocs[idx]**2 + zLocs[idx]**2) \
                / ((eps)**2))))
            Gamma_wake_rotation = 1.0 * 2 * np.pi * D * (aI - aI**2) \
                                  * turbine.average_velocity / TSR
            Gamma0 = np.mean(np.abs(Gamma))

            test_gamma = np.linspace(-30, 30, 61)
            minYaw = 10000
            for i in range(len(test_gamma)):
                tmp1 = 8 * Gamma0 / (np.pi * flow_field.air_density * D \
                       * turbine.average_velocity * Ct)
                tmp = np.abs((sind(test_gamma[i]) * cosd(test_gamma[i])**2) \
                      - tmp1)
                if tmp < minYaw:
                    minYaw = tmp
                    idx = i
            try:
                yaw_effective = test_gamma[idx]
            except:
                # TODO: should this really be handled or let the error throw
                # when this doesnt work?
                print('ERROR', idx)
                yaw_effective = 0.0

            return yaw_effective + turbine.yaw_angle

        else:
            return turbine.yaw_angle
