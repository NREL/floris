# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from .input_reader import InputReader

class Floris():
    """
    Floris is the highest level class of the Floris package. Import this class
    and instantiate it with a path to an input file to begin running Floris. Use
    the ``farm`` attribute to access other objects within the model.

    inputs:
        input_file: str - path to the json input file
    
    outputs:
        self: Floris - an instantiated Floris object
    """

    def __init__(self, input_file):

        self.input_reader = InputReader()
        self.input_file = input_file
        self.farm = self._process_input()

    def _process_input(self):
        return self.input_reader.read(self.input_file)
