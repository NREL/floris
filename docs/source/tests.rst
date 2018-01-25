
FLORIS Tests
------------

In order to maintain a level of confidence in the software, FLORIS is expected to
maintain a reasonable level of test coverage. To that end, there are unit, integration,
and regression tests included in the package.

Unit Tests
==========

Unit tests are currently included in FLORIS and integrated with the `pytest <https://docs.pytest.org/en/latest/>`_
framework. These can be executed by simply running the command
``pytest`` from the highest directory in the repository.

The currently tested modules are:

- coordinate.py

- flow_field.py

- wake.py

A testing-only class is included to provide consistent and convenient inputs 
to modules at ``sample_inputs.py``.

Integration Tests
=================
Coming soon.

Regression Tests
================
Coming soon.

Continuous Integration
======================
Continuous integration is configured with `TravisCI <https://travis-ci.org>`_ and executes all of the existing tests
for every commit. The configuration file is located in the top directory at ``.travis.yml``.

If forked, continuous integration can be included with TravisCI by simply creating an account, 
linking to a GitHub account, and turning on the switch to test the FLORIS fork.
