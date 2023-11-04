<p align="center">
<img src="https://raw.githubusercontent.com/leseixas/quasigraph/master/logo.png" style="height: 120px"></p>

[![PyPI - License](https://img.shields.io/pypi/l/quasigraph?color=green&style=for-the-badge)](LICENSE.txt)    [![PyPI](https://img.shields.io/pypi/v/quasigraph?color=red&label=version&style=for-the-badge)](https://pypi.org/project/quasigraph/) 

**Quasigraph** is an open-source toolkit designed for generating chemical and geometric descriptors to be used in machine learning models.

## Installation

The easiest method to install quasigraph is by utilizing pip:
```bash
$ pip install quasigraph
```

## Getting started

```python
from ase.build import molecule
from quasigraph import QuasiGraph

# Initialize an Atoms object for water using ASE's molecule function
atoms = molecule('H2O')

# Instantiate a QuasiGraph object containing chemical and coordination numbers
qgr = QuasiGraph(atoms)

# Convert the QuasiGraph object into a pandas DataFrame
df = qgr.to_dataframe()

# Convert the QuasiGraph object into a vector
vector = qgr.flatten()

```

## License

This is an open source code under [MIT License](LICENSE.txt).

