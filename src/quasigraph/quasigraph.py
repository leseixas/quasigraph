#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: quasigraph.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.seixas@mackenzie.br> 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Module quasigraph
'''

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from mendeleev import element
#from numba import jit

class QuasiGraph(Atoms):
    def __init__(self, atoms, pbc = False, tolerance_factor = 1.2, offset_order = 1, normalization = False, regularization = True):
        super().__init__()
        self.atoms = atoms
        self.pbc = pbc
        self.tolerance_factor = tolerance_factor
        self.normalization = normalization
        self.regularization = regularization
        if any(self.pbc):
            self.offset_order = offset_order
            self.distances_list, self.distances_tensor = self.get_distances_pbc()
            self.cn1, self.bonded_atoms = self.get_cn1_pbc()
        else:
            self.distances = self.atoms.get_all_distances()
            self.cn1, self.bonded_atoms = self.get_cn1_nopbc()
        self.cn2 = self.get_cn2()
        self.cn3 = self.get_cn3()


    def get_offsets(self):
        offset_basis = list(range(-self.offset_order,self.offset_order+1))
        offsets = [[i,j,k] for i in offset_basis for j in offset_basis for k in offset_basis]

        return offsets

    def get_distances_pbc(self):
        offsets = self.get_offsets()
        offsets_vec = [offsets[i]@self.atoms.cell for i in range(len(offsets))]
        distances_list = []
        distances_tensor = np.zeros([len(offsets), len(self.atoms), len(self.atoms)])
        for n, offset in enumerate(offsets_vec):
            for i, atom_i in enumerate(self.atoms):
                for j, atom_j in enumerate(self.atoms):
                    distance = np.linalg.norm( atom_j.position+offset - atom_i.position )
                    distances_list.append([i, j, offsets[n], distance])
                    distances_tensor[n,i,j] = distance

        return distances_list, distances_tensor 

#    @jit
    def get_cn1_nopbc(self):
        distances = self.distances
        cn1 = [0] * len(self.atoms)
        bonded_atoms = [[] for _ in range(len(self.atoms))]
        
        for i, atom_i in enumerate(self.atoms):
            CR_i = element(atom_i.symbol).covalent_radius / 100
            for j, atom_j in enumerate(self.atoms):
                CR_j = element(atom_j.symbol).covalent_radius / 100
                if i != j and distances[i, j] <= self.tolerance_factor * (CR_i + CR_j):
                    bonded_atoms[i].append(j)
                    cn1[i] += 1
                    
        return cn1, bonded_atoms

#    @jit
    def get_cn1_pbc(self):
        distances = self.distances_tensor
        cn1 = [0] * len(self.atoms)
        bonded_atoms = [[] for _ in range(len(self.atoms))]
        for n in range(len(self.get_offsets())):
            for i, atom_i in enumerate(self.atoms):
                CR_i = element(atom_i.symbol).covalent_radius / 100
                for j, atom_j in enumerate(self.atoms):
                    CR_j = element(atom_j.symbol).covalent_radius / 100
                    if 0 < distances[n,i,j] <= self.tolerance_factor * (CR_i + CR_j):
                        bonded_atoms[i].append(j)
                        cn1[i] += 1
        return cn1, bonded_atoms

    def get_cn2(self):
        cn1, bonded_atoms = self.cn1, self.bonded_atoms
        if any(self.normalization):
            norm_cn1 = max(cn1, default=1)
        else:
            norm_cn1 = 1
        cn2 = [sum(cn1[j] for j in bonded_atoms[i]) / norm_cn1 for i in range(len(self.atoms))]
        if any(self.regularization):
            cn2 = np.array(cn2) - np.array(cn1)
        return cn2

    def get_cn3(self):
        cn2, bonded_atoms = self.cn2, self.bonded_atoms
        if any(self.normalization):
            norm_cn2 = max(cn2, default=1)
        else:
            norm_cn2 = 1
        cn3 = [sum(cn2[j] for j in bonded_atoms[i]) / norm_cn2 for i in range(len(self.atoms))]
        if any(self.regularization):
            cn3 = np.array(cn3) - np.array(cn2)
        return cn3

    def get_dataframe(self):
       # Atomic data
        atoms_data = [{
            'group_id': element(atom.symbol).group_id,
            'period': element(atom.symbol).period,
            'covalent_radius': element(atom.symbol).covalent_radius / 100,
            'en_pauling': element(atom.symbol).en_pauling,
        } for atom in self.atoms]
        df = pd.DataFrame(atoms_data)

        # Geometric data
        df['CN1'] = self.cn1
        df['CN2'] = self.cn2
        df['CN3'] = self.cn3

        return df

    def get_vector(self):
        df = self.get_dataframe()
        return df.values.flatten()


if __name__ == '__main__':
  import sys
  atoms = read(sys.argv[1])
  qgr = QuasiGraph(atoms, pbc=True)
  print(qgr.get_dataframe())
