# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from pathlib import Path
from setuptools import setup, find_packages


# Package meta-data.
NAME = "FLORIS"
DESCRIPTION = "A controls-oriented engineering wake model."
URL = "https://github.com/NREL/FLORIS"
EMAIL = "rafael.mudafort@nrel.gov"
AUTHOR = "NREL National Wind Technology Center"
REQUIRES_PYTHON = ">=3.7.0"

# What packages are required for this module to be executed?
REQUIRED = [
    # simulation
    "attrs",
    "pyyaml",
    "numexpr",
    "numpy>=1.20",
    "scipy>=1.1",

    # tools
    "matplotlib>=3",
    "pandas",
    "shapely",

    # utilities
    "coloredlogs>=10.0",
]

# What packages are optional?
EXTRAS = {
    "docs": {"readthedocs-sphinx-ext", "Sphinx", "sphinxcontrib-napoleon"},
    "develop": {"pytest", "coverage[toml]", "pre-commit", "black", "isort"},
}

ROOT = Path(__file__).parent
with open(ROOT / "floris" / "VERSION") as version_file:
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
    package_data={'floris': ['VERSION']},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
)
