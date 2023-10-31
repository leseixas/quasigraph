# -*- coding: utf-8 -*-
# file: setup.py

# This code is part of QuasiGraph

from setuptools import setup, find_packages

# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(
    name = "quasigraph",
    version = "0.0.1",
    packages = find_packages(),
    author = "Leandro Seixas",
    author_email = "leandro.seixas@mackenzie.br", 
    url=" ",
    description = " ",
    long_description='''
     
    ''',
    install_requires = requirements,
    license = 'MIT',
    classifiers = [
         "Development Status :: 1 - Planning",
         "Programming Language :: Python :: 3",
         "Topic :: Scientific/Engineering :: Chemistry",
         "Topic :: Scientific/Engineering :: Physics",
         "Operating System :: OS Independent"
    ],
    python_requires = '>= 3.10.*'
)
