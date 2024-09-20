#this script need to install openlam repo and has conflict with jax due to pytorch version
#to run this script, please install a new openlam envs first and switch to the openlam envs

import sys
import glob
sys.path.append('../openlam/lam_optimize/')
from relaxer import Relaxer
from pymatgen.io.ase import AseAtomsAdaptor
import shutil, os
import ase.io
from pymatgen.core import Structure
from utils import get_e_form_per_atom
import numpy as np

def getEnergy(path, name):
    relaxer = Relaxer("mace")
    calculator = relaxer.calculator
    structure = Structure.from_file(name)
    result_temp = relaxer.relax(structure, fmax=0.05, steps=100, traj_file=None)
    atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(result_temp["final_structure"]))
    atoms.calc = relaxer.calculator
    print(get_e_form_per_atom(atoms, atoms.get_potential_energy()))
    if(get_e_form_per_atom(atoms, atoms.get_potential_energy())<-1 and get_e_form_per_atom(atoms, atoms.get_potential_energy())>-10 and np.max(abs(atoms.get_forces())) < 100):
	cif_file = f'{path}{atoms.symbols}.cif'
        ase.io.write(cif_file, atoms, format='cif')

from tqdm import tqdm
def main(path):
    name = glob.glob(f'{path}*.cif')
    os.makedirs(f'{path}excellent', exist_ok=True)
    for n in tqdm(name):
        getEnergy(path, n)

if __name__ == '__main__':
    path = sys.argv[1]
    main(path)
        
