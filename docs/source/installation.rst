
Installing FLORIS
-----------------
There are a number of ways that FLORIS can be installed. The following document
will provide instructions ranging from entry level users to contributors.

Basic Installation
==================
If you only want to run FLORIS, you can install via `pip` or `conda` with:

.. code-block:: bash
    pip install floris

OR

.. code-block:: bash
    conda install floris


Advanced Installation
=====================
If you prefer to install from source you can install via:
i.e. source code installation

.. code-block:: bash

    # master branch for the latest release version
    git clone https://github.com/NREL/floris.git -b master

    # or develop branch branch for the latest updates to the code base
    git clone https://github.com/NREL/floris.git -b develop

    # then install into your Python environment
    pip install -e floris



Developer Installation
======================
As v2.1.0, `floris` has instantiated automatic code linting and formatting that
is performed with every commit. This is accomplished through the follwing
add-ons:
- `pre-commit`
- `isort`
- `black`
- `flake8`

As such, there are a couple of extra steps involved so the following workflow
should be adhered to:
1. Clone the repository and checkout the `develop` branch:

.. code-block:: bash
    git clone https://github.com/NREL/floris.git
    cd floris/
    git checkout develop

2. Install `floris` with the developer add-ons. Pease not the quotes and dot!:

.. code-block:: bash
    pip installe -e '.[develop]'

3. Install the pre-commit workflow:
.. code-block:: bash
    pre-commit install

4. Happy developing!

OLD INSTRUCTIONS!!
==================
Installation
============
The FLORIS repository consists of two primary branches:

- `master <https://github.com/NREL/FLORIS/tree/master>`_ - Stable
  release corresponding to a specific version number.
- `develop <https://github.com/NREL/FLORIS/tree/dev>`_ - Latest
  updates including bug fixes and improvements.

These can be cloned (i.e. downloaded) directly from GitHub with one of the
following commands:

.. code-block:: bash

    # master branch
    git clone https://github.com/nrel/floris -b master

    # develop branch
    git clone https://github.com/nrel/floris -b develop

After obtaining the source code, it can be "installed" using ``pip`` or another
Python package manager. With ``pip``, there are two options:

- local editable install
- using a tagged release version from the ``pip`` repo

For consistency between all developers, it is recommended to use Python
virtual environments;
`this link <https://realpython.com/blog/python/python-virtual-environments-a-primer/>`_
provides a great introduction. Using virtual environments in a Jupyter Notebook
is described `here <https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/>`_.

Local Editable Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The local editable installation allows developers to maintain an importable
instance of FLORIS while continuing to extend it. The alternative is to
constantly update python paths within the package to match the local
environment.

Before doing the local install, the source code repository must be cloned
directly from GitHub:

.. code-block:: bash

    git clone https://github.com/nrel/floris

Then, using the local editable installation is as simple as running the
following command from the parent directory of the
cloned repository:

.. code-block:: bash

    pip install -e floris

Finally, test the installation by starting a python terminal and importing
FLORIS:

.. code-block:: bash

    import floris

pip Repo Installation
~~~~~~~~~~~~~~~~~~~~~
The FLORIS version available through the pip repository is typically the latest
tagged and released major version. This version represents the most recent
stable, tested, and validated code.

In this case, there is no need to download the source code directly. FLORIS
and its dependencies can be installed with:

.. code-block:: bash

    pip install floris

Dependencies
============
FLORIS has dependencies on various math, statistics, and plotting libraries in
addition to other general purpose packages. For the simulation and tool
modules, the dependencies are listed in ``floris/requirements.txt``. The
documentation has additional requirements listed in
``floris/docs/requirements.txt``.

The requirements files can be used to install everything with:

.. code-block:: bash

    pip install -r requirements.txt
