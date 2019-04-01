
For Developers
--------------

FLORIS is currently maintained at NREL's National Wind Technology Center by
`Jen Annoni <mailto:jennifer.annoni@nrel.gov>`_,
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_, and
`Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_. However, we are excited about
outside contribution, and this page outlines processes and procedures we'd like to follow
when contributing to the source code.

API Reference
=============
The `FLORIS API documentation <../doxygen/html/index.html>`_ is auto-generated
with Doxygen. It is a work in progress and continuously update, so please feel free to contribute!

Git and GitHub
==============
Coming soon.

Building documentation locally
==============================
This documentation is generated with Sphinx and hosted on readthedocs. However,
it can be build locally by running this command in the `docs/` directory:

::

    make html

This will create a file at `docs/build/index.html` which can be opened in any web 
browser. Note that a few dependencies are required to build the documentation locally:

- Sphinx==1.6.6
- readthedocs-sphinx-ext==0.5.17

Deploying to pip
================
Generally, only NREL developers will have appropriate permissions to deploy FLORIS updates.
When the time comes, here is a great reference on doing it:
https://medium.freecodecamp.org/how-to-publish-a-pyton-package-on-pypi-a89e9522ce24
