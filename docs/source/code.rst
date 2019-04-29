Code Reference
--------------

The `FLORIS API documentation <../doxygen/html/index.html>`_ is auto-generated
with Doxygen. It is a work in progress and continuously update, so please feel free to contribute!

FLORIS is currently maintained at NREL's National Wind Technology Center by
`Jen King <mailto:jennifer.king@nrel.gov>`_,
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_, and
`Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_. However, we are excited about
outside contribution, and this page outlines processes and procedures we'd like to follow
when contributing to the source code.

Git and GitHub
==============
Coming soon.

Building documentation locally
==============================
This documentation is generated with Sphinx and hosted on readthedocs. However,
it can be build locally by running this command in the `docs/` directory:

::

    make html

This will create a file at ``docs/build/index.html`` which can be opened in any web 
browser.

**Note** that a few additional dependencies required to build the documentation
locally are listed at ``docs/requirements.txt``.

Deploying to pip
================
Generally, only NREL developers will have appropriate permissions to deploy FLORIS updates.
When the time comes, here is a great reference on doing it:
https://medium.freecodecamp.org/how-to-publish-a-pyton-package-on-pypi-a89e9522ce24


Tests
=====

In order to maintain a level of confidence in the software, FLORIS is expected to
maintain a reasonable level of test coverage. To that end, there are unit, integration,
and regression tests included in the package.

Unit Tests
~~~~~~~~~~

Unit tests are currently included in FLORIS and integrated with the `pytest <https://docs.pytest.org/en/latest/>`_
framework. These can be executed by simply running the command
``pytest`` from the highest directory in the repository.

The currently tested modules are:

- coordinate.py

- flow_field.py

- wake.py

A testing-only class is included to provide consistent and convenient inputs 
to modules at ``sample_inputs.py``.

Regression Tests
~~~~~~~~~~~~~~~~
Coming soon.

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~
Continuous integration is configured with `TravisCI <https://travis-ci.org>`_ and executes all of the existing tests
for every commit. The configuration file is located in the top directory at ``.travis.yml``.

If forked, continuous integration can be included with TravisCI by simply creating an account, 
linking to a GitHub account, and turning on the switch to test the FLORIS fork.


.. toctree::
    :glob:
    :hidden:

    classes/floris
    classes/floris.simulation
    classes/floris.tools
