# Developer's Guide

FLORIS is maintained at NREL's National Wind Technology Center.
We are excited about community contribution, and this page outlines
processes and procedures to follow when contributing to the
source code. For technical questions regarding FLORIS usage, please
post your questions to [GitHub Discussions](https://github.com/NREL/floris/discussions).


## Git and GitHub Workflows

The majority of the collaboration and development for FLORIS takes place
in the [GitHub repository](http://github.com/nrel/floris). There,
[issues](http://github.com/nrel/floris/issues) and
[pull requests](http://github.com/nrel/floris/pulls) are managed,
questions and ideas are [discussed](https://github.com/NREL/floris/discussions),
and [new versions](http://github.com/nrel/floris/releases)
are released. It is the best mechanism for engaging with the NREL team
and other developers throughout the FLORIS community.

FLORIS development should follow "Git Flow" when interacting with the GitHub
repository. Git Flow is a git workflow outlining safe methods of pushing and
pulling commits to a shared repository. Maintaining this workflow is critical
to prevent remote changes from blocking your local development. The Git Flow
process is detailed nicely [here](http://nvie.com/posts/a-successful-git-branching-model).

### Syncing a local repository with NREL/FLORIS
The "main" FLORIS repository is continuously updated along with ongoing
research at NREL. From time to time, developers of FLORIS using their own
"local" repositories (versions of the software that exist on a local computer)
may want to sync with NREL/FLORIS. To do this, use the following git commands:

```bash
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
```

Note that the example above is a general case and may need to be modified
to fit a specific use case or purpose. If significant development has
happened locally, then [merge conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)
are likely and should be resolved as early as possible.

## Testing

In order to maintain a level of confidence in the software, FLORIS is expected
to maintain a reasonable level of test coverage. To that end, unit
tests for a the low-level code in the `floris.simulation` package are included.

The full testing suite can by executed by running the command ``pytest`` from
the highest directory in the repository. A testing-only class is included
to provide consistent and convenient inputs to modules at
`floris/tests/conftest.py`.

Unit tests are integrated into FLORIS with [pytest](https://docs.pytest.org/en/latest/),
and they can be executed with the following command:

```bash
cd floris/
pytest tests/*_unit_test.py
```

Regression tests are also included through [pytest](https://docs.pytest.org/en/latest/).
Functionally, the only difference is that the regression tests take more
time to execute and exercise a large portion of the software. These can be
executed with the following command:

```bash
cd floris/
pytest tests/*_regression_test.py
```

### Continuous Integration

Continuous integration is configured with [GitHub Actions](https://github.com/nrel/floris/actions)
and executes all of the existing tests for every push-event. The configuration file
is located at `floris/.github/workflows/continuous-integration-workflow.yaml`.

## Documentation

The online documentation is built with Jupyter Book which uses Sphinx
as a framework. It is automatically built and hosted by GitHub, but it
can also be compiled locally. Additional dependencies are required
for the documentation, and they are listed in ``docs/requirements.txt``.
The commands to build the docs are given below. After successfully
compiling, a file should be located at ``docs/_build/html/index.html``.
This file can be opened in any browser.

```bash
pip install -r docs/requirements.txt
jupyter-book build docs/

# Lots of output to the terminal here...

open docs/_build/html/index.html
```


## Deploying to pip

Generally, only NREL developers will have appropriate permissions to deploy
FLORIS updates. When the time comes, here is a great reference on doing it
is available [here](https://medium.freecodecamp.org/how-to-publish-a-pyton-package-on-pypi-a89e9522ce24).
