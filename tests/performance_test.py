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
import time

from floris.simulation import Floris
from floris.simulation import Ct, power, axial_induction, average_velocity

N_ITERATIONS = 10

def time_profile(input_dict):

    # Run once to initialize Python and memory
    floris = Floris.from_dict(copy.deepcopy(input_dict.floris))
    floris.steady_state_atmospheric_condition()

    times = np.zeros(N_ITERATIONS)
    for i in range(N_ITERATIONS):
        start = time.perf_counter()

        floris = Floris.from_dict(copy.deepcopy(input_dict.floris))
        floris.steady_state_atmospheric_condition()

        end = time.perf_counter()

        times[i] = end - start

    return np.sum(times) / N_ITERATIONS


def test_time_jensen_jimenez(sample_inputs_fixture):
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = "jensen"
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = "jimenez"
    return time_profile(sample_inputs_fixture)


def test_time_gauss(sample_inputs_fixture):
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = "gauss"
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = "gauss"
    return time_profile(sample_inputs_fixture)


def test_time_gch(sample_inputs_fixture):
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = "gauss"
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = "gauss"
    sample_inputs_fixture.floris["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.floris["wake"]["enable_secondary_steering"] = True
    sample_inputs_fixture.floris["wake"]["enable_yaw_added_recovery"] = True
    return time_profile(sample_inputs_fixture)


def test_time_cumulative(sample_inputs_fixture):
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = "cc"
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = "gauss"
    return time_profile(sample_inputs_fixture)


if __name__=="__main__":
    from conftest import SampleInputs
    print(test_time_jensen_jimenez(SampleInputs()))
    print(test_time_gauss(SampleInputs()))
    print(test_time_gch(SampleInputs()))
    print(test_time_cumulative(SampleInputs()))



"""
v3.0rc1
jan 19 2022
9e96d6c412b64fe76a57e7de8af3b00c21d18348
0.5291048929999999
0.9574160347000001
1.5500280118999996
1.3409641696

pr #56 - add CC
jan 18 2022
03e1f461c152e4f221fe92c834f2787680cf5772
0.4898268792000001
0.9306398562
1.5474471296999994
1.398449790299999

Selectively calculate Gauss near-wake and far-wake
jan 14 2022
c6bc79b0cfbc8ce5d6da0d33b68028157d2e93c0
0.4371487837000001
0.8787571535999998
1.5763470527999996
no cc model

Merge branch 'v3/unsort_yaw_angles' into redesign
jan 14 2022
8a2c1a610295c007f0222ce737723c341189811d
0.44968713739999994
0.9099841625000001
1.5407399198

Jensen: bug fix in wake masking
jan 13 2022
a325819b3b03b84bd76ad455e3f9b4600744ba14
0.44075236100000004
0.9244811726000002
1.4875763130000004

Reduce dimensions in yaw added turbulence function
jan 12 2022
66dafc08bd620d96deda7d526b0e4bfc3b086650
0.43382744339999996
0.9062629573000001
1.5413557544000003

Merge branch 'v3/ui' into redesign
jan 11 2022
12890e029a7155b074b9b325d320d1798338e287
0.43334788779999994
0.9026537795999999
1.5728437793999999

Merge branch 'v3/ui' of https://github.com/rafmudaf/floris into v3/ui
jan 10 2022
33779269e98cc882a5f066c462d8ec1eadf37a1a
0.42220622150000003
0.9103461694999998
1.5043299143000002

Put the Turbine on top level Floris
jan 6 2022
dd847210082035d43b0273ae63a76a53cb8d2e12
0.4465170094000001
0.9269890554
1.4953753718999998

Merge branch 'v3/performance' into redesign
jan 4 2022
01a02d5f91b2f4a863eebe88a618974b0749d1c4
0.43352502509999996
0.9065050287999995
1.5173905577000002

Clean up type dec module
jan 4 2022
418d8c3396c8785ea3ea56a317c5dcbea2f88fd6
0.7043072353
1.2599217953
1.8428748023000001

Remove a redundant calculation step in Jimenez
jan 3 2022
b797390a43298a815f3ff57955cfdc71ecf3e866
0.6867201590000002
1.2354249437999998
1.8026010025999994

Connect GCH components to solver
dec 29 2021
df25a9cfacd3d652361d2bd37f568af00acb2631
1.2691009706
1.2584115463999999
1.6432061797999995
"""
