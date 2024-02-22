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
from mendeleev import element
from itertools import product
# from .ptable import valence

class QuasiGraph(Atoms):
    def __init__(self, atoms, pbc = [False, False, False], tolerance = 0.4, normalization = True, show_bonded_atoms = False):
        super().__init__()
        self.atoms = atoms
        self.pbc = pbc
        self.tolerance: float = tolerance
        self.normalization: bool = normalization
        self.show_bonded_atoms: bool = show_bonded_atoms
        if any(self.pbc):
            self.offsets = [int(offset) for offset in self.pbc]
            self.distances_list, self.distances_tensor = self.get_distances_pbc()
            self.cn1, self.bonded_atoms = self.get_cn1_pbc()
        else:
            self.distances = self.atoms.get_all_distances()
            covalent_radii, positions = self.prepare_cn1_data()
            self.cn1, self.bonded_atoms = self.get_cn1_nopbc_vectorized(covalent_radii, positions)
        self.cn2 = self.get_cn2()
 

    def get_offsets(self):
        offset_basis = [list(range(-offset,offset+1)) for offset in self.offsets]
        combinations = list(product(*offset_basis))
        offsets_list = [list(comb) for comb in combinations]
        # offsets = [[i,j,k] for i in offset_basis for j in offset_basis for k in offset_basis]
        return offsets_list

    def get_distances_pbc(self):
        offsets_list = self.get_offsets()
        offsets_vec = [offsets_list[i]@self.atoms.cell for i in range(len(offsets_list))]
        # offsets_vec = [offsets[i]@self.atoms.cell for i in range(len(offsets))]
        distances_list = []
        distances_tensor = np.zeros([len(offsets_list), len(self.atoms), len(self.atoms)])
        for n, offset in enumerate(offsets_vec):
            for i, atom_i in enumerate(self.atoms):
                for j, atom_j in enumerate(self.atoms):
                    distance = np.linalg.norm( atom_j.position+offset - atom_i.position )
                    distances_list.append([i, j, offsets_list[n], distance])
                    distances_tensor[n,i,j] = distance

        return distances_list, distances_tensor 

    def prepare_cn1_data(self):
        #Store Mendeleev data in memory
        atomic_symbols = set(self.atoms.get_chemical_symbols())
        cvr = {sym: element(sym).covalent_radius for sym in atomic_symbols}

        # Compute covalent radii array
        covalent_radii = np.array([cvr[atom.symbol] / 100 for atom in self.atoms])

        # Compute positions array
        positions = np.array([atom.position for atom in self.atoms])

        return covalent_radii, positions

    def get_cn1_nopbc_vectorized(self, covalent_radii, positions):
        # Calculate distance matrix
        dist_matrix = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)

        # Create threshold matrix
        threshold_matrix = (1 + self.tolerance) * (covalent_radii[:, np.newaxis] + covalent_radii)

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
                if i != j and distances[i, j] <= (1 + self.tolerance) * (CR_i + CR_j):
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
                    if 0 < distances[n,i,j] <= (1 + self.tolerance) * (CR_i + CR_j):
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
        #Store Mendeleev data in memory
        symbols = set(self.atoms.get_chemical_symbols())
        grp = {sym: element(sym).group_id for sym in symbols}
        prd = {sym: element(sym).period for sym in symbols}
        cvr = {sym: element(sym).covalent_radius / 100 for sym in symbols}
        enp = {sym: element(sym).en_pauling for sym in symbols}
        eaf = {sym: element(sym).electron_affinity for sym in symbols}
        num_n = {sym: valence[sym][0] for sym in symbols}
        num_l = {sym: valence[sym][1] for sym in symbols}
        valc = {sym: valence[sym][2] for sym in symbols}
        
        atoms_data = [{
            'group': grp[atom.symbol],
            'period': prd[atom.symbol],
            'covalent_radius': cvr[atom.symbol],
            'en_pauling': enp[atom.symbol],
            'electron_affinity': eaf[atom.symbol],
            'num_n': num_n[atom.symbol],
            'num_l': num_l[atom.symbol],
            'valence': valc[atom.symbol]
            } for atom in self.atoms]

        df = pd.DataFrame(atoms_data)

        # Geometric data
        df['CN1'] = self.cn1
        df['CN2'] = self.cn2
        if self.show_bonded_atoms:
            df['bonded_atoms'] = self.bonded_atoms

        return df

    def get_vector(self):
        df = self.get_dataframe()
        return df.values.flatten()


if __name__ == '__main__':
  import sys
  from ase.io import read
  from ptable import valence
  atoms = read(sys.argv[1])
  qgr = QuasiGraph(atoms, pbc=[True, True, True], tolerance = 0.2 ,show_bonded_atoms=True)
  print(qgr.get_dataframe())
