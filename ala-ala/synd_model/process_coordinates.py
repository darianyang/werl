import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import numpy as np

ref_file = 'synd_model/diala_nowat_eq2.pdb'

def processCoordinates(self, coords):
    
    u_check = mda.Universe(ref_file)
    u_check.load_new(coords)
    dihed_out = []

    for frame in u_check.trajectory:
        ags = [u.residues[1].phi_selection(), u.residue[1].psi_selection()]
        results = Dihedral(ags).run().results()

        dihed_out.append(results['angles'][0])

    dihed_out = np.asarray(dihed_out)
    
    return dihed_out

    
