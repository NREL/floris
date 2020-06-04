# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import re
import sys
import time
import cProfile
from copy import deepcopy

from floris import Floris


if len(sys.argv) > 1:
    floris = Floris(sys.argv[1])
else:
    floris = Floris("example_input.json")
floris.farm.flow_field.calculate_wake()

start = time.time()


def run_floris():
    floris = Floris("example_input.json")
    return floris


cProfile.run('re.compile("floris.farm.flow_field.calculate_wake()")')
end = time.time()
print(start, end, end - start)
