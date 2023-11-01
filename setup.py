# -*- coding: utf-8 -*-
# file: setup.py

# This code is part of QuasiGraph

from pathlib import Path
from setuptools import setup, find_packages

setup(
    name = "quasigraph",
    version = "0.1.4",
    packages = find_packages(),
    author = "Leandro Seixas",
    author_email = "leandro.seixas@mackenzie.br", 
    url="https://github.com/leseixas/QuasiGraph",
    description = "QuasiGraph",
    long_description='''
    QuasiGraph
    ''',
    install_requires = [
        'numpy',
        'pandas',
        'ase',
        'mendeleev',
        'acat'
    ], 
    license = 'MIT',
    classifiers = [
         "Development Status :: 1 - Planning",
         "Programming Language :: Python :: 3",
         "Topic :: Scientific/Engineering :: Chemistry",
         "Topic :: Scientific/Engineering :: Physics",
         "Operating System :: OS Independent"
    ]
)
