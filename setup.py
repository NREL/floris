
from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = "FLORIS"
DESCRIPTION = "A controls-oriented engineering wake model."
URL = "https://github.com/NREL/FLORIS"
EMAIL = "rafael.mudafort@nrel.gov"
AUTHOR = "NREL National Wind Technology Center"
REQUIRES_PYTHON = ">=3.8.0"

# What packages are required for this module to be executed?
REQUIRED = [
    # simulation
    "attrs",
    "pyyaml~=6.0",
    "numexpr~=2.0",
    "numpy~=1.20",
    "scipy~=1.1",

    # tools
    "matplotlib~=3.0",
    "pandas~=2.0",
    "shapely~=2.0",

    # utilities
    "coloredlogs~=15.0",
    "pathos~=0.3",
]

# What packages are optional?
# To use:
#   pip install -e ".[docs,develop]"    install both sets of extras in editable install
#   pip install -e ".[develop]"         installs only developer packages in editable install
#   pip install "floris[develop]"       installs developer packages in non-editable install
EXTRAS = {
    "docs": {
        "jupyter-book",
        "sphinx-book-theme",
        "sphinx-autodoc-typehints",
        "sphinxcontrib-autoyaml",
        "sphinxcontrib.mermaid",
    },
    "develop": {
        "pytest",
        "pre-commit",
        "ruff",
        "isort",
    },
}

ROOT = Path(__file__).parent
with open(ROOT / "floris" / "version.py") as version_file:
    VERSION = version_file.read().strip()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={
        'floris': ['turbine_library/*.yaml', 'core/wake_velocity/turbopark_lookup_table.mat']
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license_files = ('LICENSE.txt',),
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
)
