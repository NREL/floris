# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

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
