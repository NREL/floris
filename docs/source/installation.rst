
Installing FLORIS
-----------------
The installation procedure for FLORIS varies depending on the inteded usage.
Developers of FLORIS will download the source code and install it in a manner
that allows running the local source code. On the other hand, those who want
to run the "released" version of FLORIS in their own script can install
with a package manager.

Basic Installation
==================
To run a released version of FLORIS, you can install via `pip` or `conda`.
By default, this installs that latest released version. To install a specific
version, specify that as shown below.

.. code-block:: bash

    # Using pip...
    pip install floris         # Latest version
    pip install floris==1.1.0  # Specified version number

    # Using conda...
    conda install floris        # Latest version
    conda install floris=1.1.0  # Specified version number


After installation, the FLORIS package can by imported in a Python
program similar to any other package.

.. code-block:: python

    import floris
    print( help( floris ) )
    print( dir( floris ) )
    print( help( floris.simulation ) )

Advanced Installation
=====================
You may prefer to download the source code of FLORIS and install
it on your system. This is required for developing FLORIS, but it
is also useful to study the software since an extensive set
of examples are provided with the repository.

The FLORIS repository consists of two primary branches:

- `master <https://github.com/NREL/FLORIS/tree/master>`_ - Stable
  release corresponding to a specific version number.
- `develop <https://github.com/NREL/FLORIS/tree/dev>`_ - Latest
  updates including bug fixes and improvements but possibly unstable.

To download the source code, use `git clone`. Then, add it to
your Python path with the "local editable install" through `pip`.

.. code-block:: bash

    # Download the source code.
    git clone https://github.com/NREL/floris.git

    # Install into your Python environment
    pip install -e floris

If everything is configured correctly, any changes made to the source
code will be available directly through your local Python. Remember
to re-import the FLORIS module when changes are made if you are working
in an interactive environment like Jupyter.

Developer Installation
======================
The FLORIS development team has included automatic code linting and
formatting utilities that are executed at every commit. This is
accomplished through the following add-ons:

- `pre-commit <https://pre-commit.com/>`_
- `isort <https://timothycrosley.github.io/isort/>`_
- `black <https://black.readthedocs.io/en/stable/>`_
- `flake8 <https://flake8.pycqa.org/en/latest/>`_

There are some extra steps required to align your development
environment with the FLORIS development team's.

Clone the repository and checkout the `develop` branch:

.. code-block:: bash

    git clone https://github.com/NREL/floris.git
    cd floris
    git checkout develop

Install FLORIS with the developer add-ons

.. important::

    You must include the two quotes (`'`) and the dot (`.`)!

.. code-block:: bash

    pip install -e '.[develop]'

Install the pre-commit utility:

.. code-block:: bash

    pre-commit install

Finally, check out :ref:`for_developers` for guidance on merging
your updates to FLORIS with the NREL repository as well as for building the
documentation locally.

Happy coding!
