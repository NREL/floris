from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='floris',
    version='0.1.0',
    description='A wind turbine wake modeling software',
    long_description=long_description,
    url='https://github.com/wisdem/floris',
    author='NREL National Wind Technology Center',
    author_email='rafael.mudafort@nrel.gov',
    license='Apache-2.0',
    classifiers=[  # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='wind turbine energy wake modeling floris nrel nwtc',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.12.1',
        'scipy >= 0.19.1',
        'matplotlib >= 2.1.0',
    ],
    python_requires='~=3.3',
)
