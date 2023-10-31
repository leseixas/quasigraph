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