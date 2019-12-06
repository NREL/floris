# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

# import matplotlib.pyplot as plt
import floris.tools as wfct

# Initialize the FLORIS interface fi
fi = wfct.floris_utilities.FlorisInterface("../example_input.json")

for i in range(100):

    # Change ws
    fi.reinitialize_flow_field(wind_speed=9)

    # Calculate wake
    fi.calculate_wake()

# # Profile reint flow
# pr = cProfile.Profile()
# pr.enable()
# for i in range(100):
#     fi.reinitialize_flow_field(wind_speed=9)
#     fi.calculate_wake()
# pr.disable()
# pr.dump_stats("result.prof")