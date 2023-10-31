#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.data import covalent_radii as CR
from mendeleev import element

class QuasiGraph(Atoms):
    def __init__(self, atoms, tolerance=0.25):
        super().__init__()
        self.atoms = atoms
        self.tolerance = tolerance
   
    def get_coordination_numbers(self):
        distances = self.atoms.get_all_distances()
        cn = []
        bonded_atoms = []
        for i, atom_i in enumerate(self.atoms):
            bondedi = []
            for j, atom_j in enumerate(self.atoms):
                if i == j:
                    continue
                if (distances[i,j] <= (1+self.tolerance)*(CR[atom_i.number]+CR[atom_j.number])):
                    bondedi.append(j)
            bonded_atoms.append(bondedi)

        for i in bonded_atoms:
            cn.append(len(i))
        return cn, bonded_atoms

    def get_callevallejo_numbers(self):
        cn, bonded_atoms = self.get_coordination_numbers()
        cvn = []
        for i, atom in enumerate(self.atoms):
            bonded_i = bonded_atoms[i]
            for j in bonded_i:
                cvn_j = np.sum(np.take(cn, bonded_i)) / max(cn)
            cvn.append(cvn_j)
        return cvn


    def to_dataframe(self):
       # Atomic data
        atoms_data = [{
            'Group': element(atom.symbol).group_id,
            'Period': element(atom.symbol).period,
            'Covalent radius': element(atom.symbol).covalent_radius,
            'Electronegativity': element(atom.symbol).en_pauling,
        } for atom in self.atoms]
        df = pd.DataFrame(atoms_data)

        # Geometric data
        cn, bonded_atoms = self.get_coordination_numbers()
        cvn = self.get_callevallejo_numbers()
        df['CN'] = cn
        df['CVN'] = cvn

        return df

    def flatten(self):
        df = self.to_dataframe()
        return df.to_numpy().flatten()


if __name__ == '__main__':
  atoms = read(sys.argv[1])
  gcoord = QuasiGraph(atoms).to_dataframe()
  print(gcoord)
#  for i in gcoord:
#    print(i)
