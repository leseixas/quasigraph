<p align="center">
<img src="https://raw.githubusercontent.com/leseixas/quasigraph/master/resources/logo.png" style="height: 150px"></p>

[![PyPI - License](https://img.shields.io/pypi/l/quasigraph?color=green&style=for-the-badge)](LICENSE.txt)    [![PyPI](https://img.shields.io/pypi/v/quasigraph?color=red&label=version&style=for-the-badge)](https://pypi.org/project/quasigraph/) 

**Quasigraph** is an open-source toolkit designed for generating chemical and geometric descriptors to be used in machine learning models.

# Installation

The easiest method to install quasigraph is by utilizing pip:
```bash
$ pip install quasigraph
```

# Getting started

```python
from ase.build import molecule
from quasigraph import QuasiGraph

# Initialize an Atoms object for water using ASE's molecule function
atoms = molecule('CH3OH')

# Instantiate a QuasiGraph object containing chemical and coordination numbers
qgr = QuasiGraph(atoms)

# Convert the QuasiGraph object into a pandas DataFrame
df = qgr.get_dataframe()

# Convert the QuasiGraph object into a vector
vector = qgr.get_vector()
```

# Descriptor

The descriptor can be separated into two parts, a chemical part and a geometric part.

## Chemical part

The chemical part of the descriptor employs the [Mendeleev library](https://github.com/lmmentel/mendeleev), incorporating atomic details like the group, period, covalent radius, and Pauling electronegativity for every element within the object.

For example, for methanol (CH<sub>3</sub>OH) we have the table:

|    |   group_id |   period |   covalent_radius |   en_pauling |
|---:|-----------:|---------:|------------------:|-------------:|
|  0 |         14 |        2 |              0.75 |         2.55 |
|  1 |         16 |        2 |              0.63 |         3.44 |
|  2 |          1 |        1 |              0.32 |         2.2  |
|  3 |          1 |        1 |              0.32 |         2.2  |
|  4 |          1 |        1 |              0.32 |         2.2  |
|  5 |          1 |        1 |              0.32 |         2.2  |

## Geometric part

The geometric part involves identifying all bonds and computing the coordination numbers for each atom, indicated as CN1. Additionally, the generalized coordination number (CN2)[^1] is determined by summing the coordination numbers of the neighboring ligands for each atom and normalizing this sum by the highest coordination number found in the molecule.

```{figure} https://raw.githubusercontent.com/leseixas/quasigraph/master/resources/methanol.png
:name: Figure1
:align: center

Schematic representation of the methanol molecule, indicating the chemical symbol and coordination number (CN1) for every atom.
```

For example, for methanol (CH<sub>3</sub>OH) we have the geometric data, as shown in [Fig. 1](#fig1).

|   CN1 |   CN2 |
|------:|------:|
|     4 |  1.25 |
|     2 |  1.25 |
|     1 |  1.00 |
|     1 |  0.50 |
|     1 |  1.00 |
|     1 |  1.00 |

# License

This is an open source code under [MIT License](LICENSE.txt).

# Acknowledgements

We thank financial support from FAPESP (Grant No. 2022/14549-3), INCT Materials Informatics (Grant No. 406447/2022-5), and CNPq (Grant No. 311324/2020-7).

# References

[^1]: Calle-Vallejo, F., Martínez, J. I., García-Lastra, J. M., Sautet, P. & Loffreda, D. [Fast Prediction of Adsorption Properties for Platinum Nanocatalysts with Generalized Coordination Numbers](https://doi.org/10.1002/anie.201402958), *Angew. Chem. Int. Ed.* **53**, 8316-8319 (2014).