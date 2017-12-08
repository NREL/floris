## FLORIS

The `develop` branch of the FLORIS repo represents a rework of the software as a general wind farm wake analysis tool.

For any questions regarding FLORIS, please contact [Jen Annoni](mailto:jennifer.annoni@nrel.gov), [Paul Fleming](mailto:paul.fleming@nrel.gov), or [Rafael Mudafort](mailto:rafael.mudafort@nrel.gov).

**NOTE: This FLORIS repository is under active development. Be aware that the implementation is not yet validated or verified.**

### Background and objectives
Coming soon.

### Architecture
An architecture diagram as in input to [draw.io](https://www.draw.io) is contained in the repository at `FLORIS/florisarch.xml`.

Generally, a user will not have to write Python code in order to express a wind farm, turbine, or wake model combination. Currently, however,
specific wind farms, turbines, and wake model combinations are introduced through classes in `FLORIS/farms`, `FLORIS/turbines`, and `FLORIS/wakes`, but
a more robust and code free input process is coming soon.

### Download
FLORIS can be cloned directly from GitHub:  
``git clone -b develop https://github.com/wisdem/floris``

### Dependencies
The following packages are required for FLORIS
- Python3
- NumPy v1.12.1
- SciPy v0.19.1
- matplotlib v2.1.0

After installing Python3, the remaining dependencies can be installed with `pip` referencing the requirements list using this command:  
`pip install -r requirements.txt`

### Executing FLORIS
Currently, FLORIS is executed by simply running `FLORIS.py`:  
`python3 FLORIS.py`

### Future work
- improve the swept area point sampling to consistently include more points
