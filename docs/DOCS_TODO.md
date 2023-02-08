# FLORIS Documentation -> Diataxis Framework Notes/Commentary/To Do Items

## Landing Page

- Currently, this is overwhelming from a non-legacy/new user standpoint.
- Keep
  - The intro paragraph and titles
- Suggested edits
  - Add current versioning, copyright, licensing, and some high level tidbits, which can also be done through badges I'm looking at the pandas documentaion currently and thinking about how it doesn't dive straight into the code or examples on the landing pages so users can navigate to the kinds of material they are looking for).
  - Maybe use some of the produced figures from studies or examples to help guide users around to what they might be looking for

## Left Sidebar

- Each of the main sections, such as Background, Examples/Turorials, API, and Getting Started should have separate sections, aka, a line separating those content pieces with the main headers underneath (thinking about the Myst-NB or Myst-Parser documentation here)

## Alternatively, the Top Bar for PyData Sphinx

- This could be used for the four main categories to go to completely different sites with a separate left/right sidebar in each for the top-level subheadings, and each section's subheadings (for example: Install on the left and Pip, Source, Developer, etc. on the right)

# Installation

- I made some changes to delineate the different considerations into subsections

# Tutorials/How-Tos and Examples

- It seems like users might benefit from bringing the *Background and Concepts*, *Examples Index*, *Input File Reference* under one umbrella to put them together in easier to follow series of guides
-
## Background and Concepts

- This is definitely a tutorial, which is fine, but I think just needs to be renamed to something more like *Beginner's Guide* or *FLORIS 101*
- I assumed this was going to be the background on wake modeling or how we got FLORIS, and it might be useful to have that kind of section, which I believe existed in the original documentation.
- Should the demonstration of the array dimensions and axes be something more like the following in place of what seems like the creation of a new array? I found this to be slightly confusing at first glance, and thought it might be helpful for people less well-versed in NumPy.
  - 0: number of wind directions
  - 1: number of wind speeds
  - 2: number of turbines
  - 3: grid-1, or number of horizontal grid points on the rotor,
  - 4: grid-2, or number of vertical grid points on the rotor,
- In *Execute wake calculation*, I think we should say "Running the wake calculation is a single method call" because a one-liner is often in reference to a longer computation condensed into a single line of code
- It seems like it'd be useful to clump the initialization step through yaw angle step into a single "Basic Workflow" heading, ending with a putting it all together example, then move onto the next set of "Advanced Workflow" or "Other Analysis Steps" to then add the other considerations. I think this might be nice for helping to split up the large amount of content in here a little bit more cleanly.
- In a similar vein, the Visualization examples could get their own subsection just to clearly delineate the themes and create cleaner breaks in the content.
- The *On Grid Points* section seems to be a repeat of an earlier step in *Build the model*, and so the written content could be moved up there

## Input File Reference

- I like the use of the dictionary bullets to explain everything.
- The only things that I'd add here are:
  - a sentence or two to explain the motivation of this page to help transition from the previous content into the file defintion content
  - an example file at the end with an initialization step, just to show it working.

## Examples Index

- This is likely a decent chunk of work, but I'd recommend turning the example scripts into myst-nb compatible markdown that can generate the .ipynb and .py files.
- I like how each example file has a high level summary of what's covered, and definitely think we should keep that.
- What would be nice is if we could view the contents of the example in the documentation, via myst-nb.
  - Automating this step would allow us to also host examples on Binder and direct users there to get started.

# Developer's Guide

- There is no one step to get running for installation, which makes this a little bit confusing because a lot of the other content is useful, but needs a little motivation. The following section flow is an example of what could be helpful
  - Intro paragraph
  - Getting started: fork, clone, pip install, and pre-commit install as follows, so that each component can be later referenced, but at least there is a workflow to start from, and it's at the beginning of the guide.
  ```bash
  # Create your own fork
  <the code>

  # Clone the repo
  <the code>

  # Install FLORIS in editable mode with one or both options in the squuare brackets
  # depending on how you plan to contribute code and/or documentation
  pip install -e "floris[develop, docs]"

  # Enable the code quality tools
  pre-commit install
  ```
  - Git/GitHub workflow



# FlorisInterface and Tools

- How we use these tools and how should we do this?
- What should this API be? Does it need a renaming?
- Most importantly, what should FlorisInterface look like?
- Do we need to reset the analysis at the end?
  - Reduces the ability to pull intermediary calculations
  -
