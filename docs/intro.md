# FLORIS Wake Modeling & Wind Farm Controls

FLORIS is a controls-focused wind farm simulation software incorporating
steady-state engineering wake models into a performance-focused Python
framework.
The software is in active development and engagement with the development team
is highly encouraged. If you are interested in using FLORIS to conduct studies
of a wind farm or extending FLORIS to include your own wake model, please join
the conversation in [GitHub Discussions](https://github.com/NREL/floris/discussions/)!

## Quick Start

FLORIS is a Python package run on the command line typically by providing
an input file with an initial configuration. It can be installed with
```pip install floris``` (see [](installation)). The typical entry point is
{py:class}`.FlorisInterface` which accepts the path to the
input file as an argument. From there, changes can be made to the initial
configuration through the {py:meth}`.FlorisInterface.reinitialize`
routine, and the simulation is executed with
{py:meth}`.FlorisInterface.calculate_wake`.

```python
from floris.tools import FlorisInterface
fi = FlorisInterface("path/to/input.yaml")
fi.reinitialize(wind_directions=[i for i in range(10)])
fi.calculate_wake()
```

Finally, results can be analyzed via post-processing functions avilable within
{py:class}`.FlorisInterface` such as
{py:meth}`.FlorisInterface.get_turbine_layout`,
{py:meth}`.FlorisInterface.get_turbine_powers` and
{py:meth}`.FlorisInterface.get_farm_AEP`, and
a visualization package is available in {py:mod}`floris.tools.visualization`.
A collection of examples are included in the [repository](https://github.com/NREL/floris/tree/main/examples)
and described in detail in [](examples).

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/floris/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/floris/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/18/): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/floris/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/floris/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/floris/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.
