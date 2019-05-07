Change Log
----------

For download and installation instructions, see the
:ref:`installation <installation>` section.

v1.0.0
======
Released on 2019-05-07

**This update introduces breaking changes to the FLORIS API.**
See the `code reference documentation <https://floris.readthedocs.io/en/develop/source/code.html>`__
for detailed information.

- Adds the Curl wake model (Martinez-Tossas et al)
- Improves the FLORIS object hierarchy
- Incorporates an analysis tools package alongside the simulation package
- Adds examples demonstrating using the ``simulation`` and ``tools`` packages
- Adds in-code documentation via doc strings
- Reworks and expands the online documentation

v0.4.0
======
Released on 2018-10-19

- Adds a web app for generating input files
- Improves visualization
- Documentation updates
- Improves the module naming and general project organization

v0.3.1
======
Released on 2018-04-03

- Bug fix in unit tests

v0.3.0
======
Released on 2018-04-03

- Connects the air density input to the models
- Improves the documentation and readme
- Moves the visualization tools out of the main package
- Adds examples on AEP calculation

v0.2.0
======
Released on 2018-03-20

- Adds user-specified turbulence intensity parameters to the gaussian model
- Uses a smaller grid for general wake calculations
- Adds visualization support in post processing
- Bug fixes for the example scripts
- Updates and improved the documentation
- Improves module imports
- Moves the tests directory in /floris to avoid global namespace conflicts

v0.1.1
======
Released on 2018-01-25

- Removes the requirement for unit tests to pass in Floris instantiation

v0.1.0
======
Released on 2018-01-15

- Initial release of FLORIS
