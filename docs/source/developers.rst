For Developers
--------------
FLORIS is currently maintained at NREL's National Wind Technology Center by
`Jen King <mailto:jennifer.king@nrel.gov>`_,
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_,
`Chris Bay <mailto:chris.bay@nrel.gov>`_, and
`Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_. However, we are excited
about community contribution, and this page outlines processes and procedures
we'd like to follow when contributing to the source code.

Git and GitHub
==============
The majority of the collaboration and development for FLORIS takes place
in the `GitHub repository <http://github.com/nrel/floris>`__. There,
`issues <http://github.com/nrel/floris/issues>`__ and
`pull requests <http://github.com/nrel/floris/pulls>`__
are discussed and new versions are released. It is the best mechanism for
engaging with the NREL team and other developers throughout
the FLORIS community.

FLORIS development should follow "Git Flow" when interacting with the GitHub
repository. Git Flow is a git workflow outlining safe methods of pushing and
pulling commits to a shared repository. Maintaining this workflow is critical
to prevent remote changes from blocking your local development. The Git Flow
process is detailed nicely
`here <http://nvie.com/posts/a-successful-git-branching-model>`__.

Building Documentation Locally
==============================
This documentation is generated with Sphinx and hosted on readthedocs. However,
it can be built locally by running this command in the ``floris/docs/``
directory:

.. code-block:: bash

    make html

This will create a file at ``floris/docs/_build/html/index.html`` which
can be opened in any web browser.

**Note** that a few additional dependencies required to build the documentation
locally are listed at ``floris/docs/requirements.txt``.

Testing
=======

In order to maintain a level of confidence in the software, FLORIS is expected
to maintain a reasonable level of test coverage. To that end, there are unit
and regression tests included in the package.

The full testing suite can by executed by running the command ``pytest`` from
the highest directory in the repository. A testing-only class is included
to provide consistent and convenient inputs to modules at
``floris/tests/sample_inputs.py``.

Unit Tests
~~~~~~~~~~

Unit tests are integrated into FLORIS with the
`pytest <https://docs.pytest.org/en/latest/>`_ framework. These can be executed
by running the command ``pytest tests/*_unit_test.py`` from the highest
directory in the repository.

Regression Tests
~~~~~~~~~~~~~~~~
Regression tests are included in FLORIS through the same
`pytest <https://docs.pytest.org/en/latest/>`_ framework as the unit tests.
Functionally, the only difference is that the regression tests take more
time to execute and exercise a large portion of the software. These can be
executed by running the command ``pytest tests/*_regression_test.py`` from the
highest directory in the repository.

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~
Continuous integration is configured with `TravisCI <https://travis-ci.org/NREL/floris>`_
and executes all of the existing tests for every commit. The configuration file
is located in the top directory at ``floris/.travis.yml``.

If forked, continuous integration can be included with TravisCI by simply
creating an account, linking to a GitHub account, and turning on the switch to
test the FLORIS fork.

Deploying to pip
================
Generally, only NREL developers will have appropriate permissions to deploy
FLORIS updates. When the time comes, here is a great reference on doing it:
https://medium.freecodecamp.org/how-to-publish-a-pyton-package-on-pypi-a89e9522ce24
