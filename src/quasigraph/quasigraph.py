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
# from numba import jit

class QuasiGraph(Atoms):
    def __init__(self, atoms, pbc = False, tolerance_factor = 1.2, offset_order = 1, normalization = True):
        super().__init__()
        self.atoms = atoms
        self.pbc = pbc
        self.tolerance_factor = tolerance_factor
        self.normalization = normalization
        if any(self.pbc):
            self.offset_order = offset_order
            self.distances_list, self.distances_tensor = self.get_distances_pbc()
            self.cn1, self.bonded_atoms = self.get_cn1_pbc()
        else:
            self.distances = self.atoms.get_all_distances()
            covalent_radii, positions = self.prepare_cn1_data()
            self.cn1, self.bonded_atoms = self.get_cn1_nopbc_vectorized(covalent_radii, positions)
            #self.cn1, self.bonded_atoms = self.get_cn1_nopbc()
        self.cn2 = self.get_cn2()
 

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

    def prepare_cn1_data(self):
        # Compute covalent radii array
        covalent_radii = np.array([element(atom.symbol).covalent_radius / 100 for atom in self.atoms])

        # Compute positions array
        positions = np.array([atom.position for atom in self.atoms])

        return covalent_radii, positions

    def get_cn1_nopbc_vectorized(self, covalent_radii, positions):
        # Calculate distance matrix
        dist_matrix = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)

        # Create threshold matrix
        threshold_matrix = self.tolerance_factor * (covalent_radii[:, np.newaxis] + covalent_radii)

        # Compare distances with thresholds
        bonding_matrix = (dist_matrix <= threshold_matrix)

        # Set diagonal to False to ignore self-bonding
        np.fill_diagonal(bonding_matrix,False)

        # Count bonded atoms
        cn1 = bonding_matrix.sum(axis=1)

        # Determine bonded atoms
        bonded_atoms = [list(np.where(row)[0]) for row in bonding_matrix]

        return cn1, bonded_atoms

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
        if self.normalization:
            norm_cn1 = max(cn1, default=1)
        else:
            norm_cn1 = 1
        cn2 = [sum(cn1[j] for j in bonded_atoms[i]) / norm_cn1 for i in range(len(self.atoms))]
        return cn2

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

        return df

    def get_vector(self):
        df = self.get_dataframe()
        return df.values.flatten()


if __name__ == '__main__':
  # import sys
  # atoms = read(sys.argv[1])
  from ase.cluster import Icosahedron
  atoms = Icosahedron("Pt", noshells=3)
  qgr = QuasiGraph(atoms, pbc=False)
  print(qgr.get_vector())
