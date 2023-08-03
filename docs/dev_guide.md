# Developer's Guide

FLORIS is maintained at NREL's National Wind Technology Center.
We are excited about community contribution, and this page outlines
processes and procedures to follow when contributing to the
source code. For technical questions regarding FLORIS usage, please
post your questions to [GitHub Discussions](https://github.com/NREL/floris/discussions).

## Getting Started

There are a few steps that nearly all contributors will need to go through to get started with
making contributions to FLORIS. Each of these steps will be addressed in a later section of the
developer's guide, so please read on to learn more about each of these steps.

1. Create a fork of FLORIS on GitHub
2. Clone the repository

    ```bash
    git clone -b develop https://github.com/<your-GitHub-username>/floris.git
    ```

3. Move into the FLORIS source code directory

    ```bash
    cd floris/
    ```

4. Install FLORIS in editable mode with the appropriate developer tools

   - ``".[develop]"`` is for the linting and code checking tools
   - ``".[docs]"`` is for the documentation building tools. Ideally, developers should also be
     contributing to the documentation, and therefore checking that the documentation builds locally.

    ```bash
    pip install -e ".[develop, docs]"
    ```
5. Turn on the linting and code checking tools
   ```bash
   pre-commit install
   ```

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


## Code quality tools

FLORIS is configured to use tools to automatically check and enforce
aspects of code quality. In general, these should be adopted by all
developers and incorporated into the development workflow. Most
tools are configured in [pyproject.toml](https://github.com/NREL/floris/blob/main/pyproject.toml),
but some may have a dedicated configuration file.

### isort

Import lines can easily get out of hand and cause unnecessary distraction
in source code files. [isort](https://pycqa.github.io/isort/index.html)
is used to automatically manage imports in the source code. It can be run
directly with the following command:

```bash
isort <path to file>
isort dir/*
```

This tool was initially configured in [PR#535](https://github.com/NREL/floris/pull/535),
and additional information on specific decisions can be found there.

### Ruff

[Ruff](https://github.com/charliermarsh/ruff) is a general linter and limited auto-formatter.
It is configured in `pyproject.toml` through various `[tool.ruff.*]` blocks. It is a command line
tool and integrations into popular IDE's are available. A typical command to run Ruff for all of
FLORIS is the following:

```bash
ruff . --fix
```

This sets the configuration from `pyproject.toml`, applies the selected rules to Python files,
and fixes errors in-place where possible.

Ruff was initially configured in [PR#562](https://github.com/NREL/floris/pull/562), and discussed
in more detail in [D#561](https://github.com/NREL/floris/discussions/561). See the Ruff
documentation for a list of [supported rules](https://github.com/charliermarsh/ruff#supported-rules)
and [available options for various rules](https://github.com/charliermarsh/ruff#reference).

### Pre-commit

[Pre-commit](https://pre-commit.com) is a utility to execute a series of git-hooks to catch
and fix formatting errors. It integrates easily with other tools, in our case isort and Ruff.
Pre-commit is tightly integrated into git and is mostly not used directly by users.

Once installed, the precommit hooks must be installed into each development environment with
the following command from the `floris/` directory:

```bash
pre-commit install
```

Then, each commit creation with `git` will run the installed hooks and display where
checks have failed. Pre-commit will typically modify the files directly to fix the issues. However,
each file fixed by Pre-commit must be added to the git-commit again to capture the changes. A
typical workflow is given below.

```bash
git add floris/simulation/turbine.py
git commit -m "Update so and so"
> [WARNING] Unstaged files detected.
> [INFO] Stashing unstaged files to /Users/rafmudaf/.cache/pre-commit/patch1675722485-25489.
> isort....................................................................Failed
> - hook id: isort
> - files were modified by this hook
>
> Fixing /Users/rafmudaf/Development/floris/floris/simulation/turbine.py

# Check that the error is fixed or fixed it manually
git status

# Stage the new changes and commit
git add floris/simulation/turbine.py
git commit -m "Update so and so"
```

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
for the documentation, and they are listed in the `EXTRAS` of `setup.py`.
The commands to build the docs are given below. After successfully
compiling, a file should be located at ``docs/_build/html/index.html``.
This file can be opened in any browser.

```bash
pip install -e ".[docs]"
jupyter-book build docs/

# Lots of output to the terminal here...

open docs/_build/html/index.html
```

## Deploying to pip

Generally, only NREL developers will have appropriate permissions to deploy
FLORIS updates. When the time comes, here is a great reference on doing it
is available [here](https://medium.freecodecamp.org/how-to-publish-a-pyton-package-on-pypi-a89e9522ce24).
