
Examples
---------

The FLORIS code includes wake models, and a number of related analysis and visualization tools to be used in
connection with wind farm controls research.  A number of examples are provided in the directory ``examples/``
to provide instruction on the use of most of the underlying codes.

.. toctree::
    :glob:
    
    examples/example_0000.rst
    examples/example_0005.rst
    examples/example_0006.rst
    examples/example_0007.rst
    examples/example_0010.rst
    examples/example_0010a.rst
    examples/example_0011.rst
    examples/example_0011a.rst
    examples/example_0012.rst
    examples/example_0015.rst
    examples/example_0020.rst
    examples/example_0030.rst
    examples/example_0040.rst

For questions not covered in the examples, or to request additional examples, please first search for or 
submit your questions to stackoverflow.com using the tag FLORIS.  Additionally you can contact
the NREL FLORIS team at `NREL.Floris@nrel.gov <mailto:floris@nrel.gov>`_ or
`Jen King <mailto:jennifer.king@nrel.gov>`_ and
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_ directly.

FLORIS Input
============
A sample input file to the Floris model is provided :ref:`here <sample_input_file_ref>`.
This example case uses the NREL 5MW turbine and the Gaussian wake model as a reference.
All model parameters provided have been published in previous work, but the inputs to
the example input file can be changed as needed. However, be aware that changing these parameters
may result in an unphysical solution.  Many of the example files will make use of this example input.


License
=======

Copyright 2019 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.