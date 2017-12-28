"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from .InputReader import InputReader


class floris():
    def __init__(self):
        if test_floris().fails():
            error = "floris unit tests failed. " \
                + "Run the standalone pytest framework to debug."
            raise RuntimeError(error)
        
        self.input_reader = InputReader()

    def process_input(self, input_file):
        self.farm = self.input_reader.read(input_file)

class test_floris():
    def __init__(self):
        pass

    def fails(self):
        return False
