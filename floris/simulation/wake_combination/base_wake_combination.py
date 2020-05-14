# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...logging_manager import LoggerBase


class WakeCombination(LoggerBase):
    """
    This is the super-class for all combination models included in FLORIS.
    These models define how the wake velocity deficits are combined with each
    other and the freestream velocity field.
    """

    def __init__(self):
        self.model_string = None
