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
        coordination_numbers = [0] * len(self.atoms)
        bonded_atoms = [[] for _ in range(len(self.atoms))]
        
        for i, atom_i in enumerate(self.atoms):
            for j, atom_j in enumerate(self.atoms):
                if i != j and distances[i, j] <= (1 + self.tolerance) * (CR[atom_i.number] + CR[atom_j.number]):
                    bonded_atoms[i].append(j)
                    coordination_numbers[i] += 1
                    
        return coordination_numbers, bonded_atoms

## Old version
#  def get_coordination_number(self):
#        distances = self.atoms.get_all_distances()
#        cn = []
#        bonded_atoms = []
#        for i, atom_i in enumerate(self.atoms):
#            bondedi = []
#            for j, atom_j in enumerate(self.atoms):
#                if i == j:
#                    continue
#                if (distances[i,j] <= (1+self.tolerance)*(CR[atom_i.number]+CR[atom_j.number])):
#                    bondedi.append(j)
#            bonded_atoms.append(bondedi)
#
#        for i in bonded_atoms:
#            cn.append(len(i))
#        return cn, bonded_atoms

    def get_callevallejo_numbers(self):
        coordination_numbers, bonded_atoms = self.get_coordination_numbers()
        max_coordination_number = max(coordination_numbers, default=1)
        cvn = [sum(coordination_numbers[j] for j in bonded_atoms[i])  / max_coordination_number for i in range(len(self.atoms))]

## Old version
#   def get_callevallejo_numbers(self):
#        cvn = [np.sum(coordination_numbers[j] for j in bonded_atoms[i]) / max_coordination_number for i in range(len(self.atoms))]
#        cn, bonded_atoms = self.get_coordination_numbers()
#        cvn = []
#        for i, atom in enumerate(self.atoms):
#            bonded_i = bonded_atoms[i]
#            for j in bonded_i:
#                cvn_j = np.sum(np.take(cn, bonded_i)) / max(cn)
#            cvn.append(cvn_j)
        return cvn


    def to_dataframe(self):
       # Atomic data
        atoms_data = [{
            'group_id': element(atom.symbol).group_id,
            'period': element(atom.symbol).period,
            'covalent_radius': element(atom.symbol).covalent_radius,
            'en_pauling': element(atom.symbol).en_pauling,
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
        return df.values.flatten()


if __name__ == '__main__':
  atoms = read(sys.argv[1])
  qg_atoms = QuasiGraph(atoms)
#  print(qg_atoms.flatten())
  print(qg_atoms.to_dataframe())
