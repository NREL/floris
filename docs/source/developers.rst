.. _for_developers:

For Developers
--------------
FLORIS is maintained at NREL's National Wind Technology Center.
We are excited about community contribution, and this page outlines
processes and procedures we'd like to follow when contributing to the
source code.

For technical questions regarding FLORIS usage please post your questions to
`GitHub Discussions <https://github.com/NREL/floris/discussions>`_ on the
FLORIS repository. We no longer plan to actively answer questions on
StackOverflow and will use GitHub Discussions as the main forum for FLORIS.
Alternatively, email the NREL FLORIS team at
`christopher.bay@nrel.gov <mailto:christopher.bay@nrel.gov>`_,
`bart.doekemeijer@nrel.gov <mailto:bart.doekemeijer@nrel.gov>`_,
`rafael.mudafort@nrel.gov <mailto:rafael.mudafort@nrel.gov>`_, or
`paul.fleming@nrel.gov <mailto:paul.fleming@nrel.gov>`_.

Git and GitHub
==============
The majority of the collaboration and development for FLORIS takes place
in the `GitHub repository <http://github.com/nrel/floris>`__. There,
`issues <http://github.com/nrel/floris/issues>`__ and
`pull requests <http://github.com/nrel/floris/pulls>`__
are discussed and `new versions <http://github.com/nrel/floris/releases>`__
are released. It is the best mechanism for
engaging with the NREL team and other developers throughout
the FLORIS community.

FLORIS development should follow "Git Flow" when interacting with the GitHub
repository. Git Flow is a git workflow outlining safe methods of pushing and
pulling commits to a shared repository. Maintaining this workflow is critical
to prevent remote changes from blocking your local development. The Git Flow
process is detailed nicely
`here <http://nvie.com/posts/a-successful-git-branching-model>`__.

Syncing a local repository with NREL/FLORIS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The "main" FLORIS repository is continuously updated along with ongoing
research at NREL. From time to time, developers of FLORIS using their own
"local" repositories (versions of the software that exist on a local computer)
may want to sync with NREL/FLORIS. To do this, use the following git commands:

.. code-block:: bash

    # Move into the FLORIS source code directory;
    # this may be named differently on your computer.
    cd floris/

    # Find the remote name that corresponds to
    # NREL/FLORIS; usually "origin" or "upstream".
    git remote -v

    # Fetch the changes on all remotes.
    git fetch --all

    # Decide which branch to sync with
    # NREL/FLORIS. Generally, this will be "main".
    git checkout main
    git pull origin main

    # Update any local working branches with the
    # latest from NREL/FLORIS.
    git checkout feature/working_branch
    git merge main

Note that the example above is a general case and may need to be modified
to fit a specific use case or purpose. If significant development has
happened locally, then `merge conflicts <https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts>`__
are likely and should be resolved as soon as possible.

Building Documentation Locally
==============================
This documentation is generated with Sphinx and hosted on readthedocs. However,
it can be built locally on any computer with Python installed which is especially
helpful when modifying the documentation. Some additional dependencies
are required for the documentation. See the commands below to install
dependencies and build the documentation.

.. code-block:: bash

    # Move into the docs directory
    cd floris/docs

    # Install documentation-specific dependencies
    pip install -r requirements.txt

    # List all available commands
    make help

    # Build the HTML-based documentation
    make html

This will create a file at ``floris/docs/_build/html/index.html`` that
can be opened in any web browser.

Testing
=======

In order to maintain a level of confidence in the software, FLORIS is expected
to maintain a reasonable level of test coverage. To that end, unit
tests for a small subset of the `simulation` package are included.

The full testing suite can by executed by running the command ``pytest`` from
the highest directory in the repository. A testing-only class is included
to provide consistent and convenient inputs to modules at
``floris/tests/conftest.py``.

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
Continuous integration is configured with `GitHub Actions <https://github.com/nrel/floris/actions>`_
and executes all of the existing tests for every push event. The configuration file
is located at ``floris/.github/workflows/continuous-integration-workflow.yaml``.

Deploying to pip
================
Generally, only NREL developers will have appropriate permissions to deploy
FLORIS updates. When the time comes, here is a great reference on doing it:
https://medium.freecodecamp.org/how-to-publish-a-pyton-package-on-pypi-a89e9522ce24
