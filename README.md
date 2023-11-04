# quasigraph

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

# Instantiate a QuasiGraph object with the Atoms object
qg = QuasiGraph(atoms)

# Convert the QuasiGraph data into a pandas DataFrame containing chemical and coordination number details
df = qg.to_dataframe()

```

## License

This is an open source code under [MIT License](LICENSE.txt).

