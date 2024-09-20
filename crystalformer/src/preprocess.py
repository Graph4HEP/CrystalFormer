import zipfile
import tarfile
import random
import jax
from glob import glob
import jax.numpy as jnp
import json
import sys,os
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

sys.path.append('crystalformer/src/')
from wyckoff import mult_table
from elements import element_list

def load_data_from_path(path):
    cif = []
    for f in glob(path):
        cif_data = load_data_from_tar(f)
        cif += cif_data
    return cif

def load_data_from_tar(tar_gz_filename):
    print(f"loading data from {tar_gz_filename}...")
    cif_data = []
    with tarfile.open(tar_gz_filename, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc=f"extracting files...  {tar_gz_filename}"):
            f = tar.extractfile(member)
            if f is not None:
                try:
                    content = f.read().decode("utf-8")
                except:
                    continue
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".cif", "")
                cif_data.append({'filename':cif_id, 'cif':content})
    return cif_data


def read_zip_file(path):
    cif = []
    with zipfile.ZipFile(path, 'r') as f:
        total_files = len(f.infolist())
        count = 0
        for info in tqdm(f.infolist(), total=total_files, desc="Processing files"):            
            if(count==0):
                count+=1
                continue
            name = info.filename
            file_content = f.read(info).decode('utf-8')

            cif.append({
                    'filename': name,
                    'cif': file_content,                    
            })
            count+=1
    return cif

def read_csv_file(path):
    cif_temp = pd.read_csv(path)['cif']
    cif = []
    for c in cif_temp:
        cif.append({'filename':path, 'cif':c})
    return cif

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def sort_atoms(W, A, X):
    """
    lex sort atoms according W, X, Y, Z

    W: (n, )
    A: (n, )
    X: (n, dim) int
    """
    W_temp = np.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort

    X -= np.floor(X)
    idx = np.lexsort((X[:,2], X[:,1], X[:,0], W_temp))

    A = A[idx]
    X = X[idx]
    return A, X

def process_cif(cif, p):
    lst = []
    for cc in tqdm(cif, total=len(cif), desc=f'{p} Processing files', mininterval=1):
        file_content = cc['cif']
        try:
            crystal = Structure.from_str(file_content, fmt='cif')
        except:
            continue
        try:
            spga = SpacegroupAnalyzer(crystal, symprec=0.001)
        except:
            try:
                spga = SpacegroupAnalyzer(crystal, symprec=0.0001)
            except:
                continue
        try:
            crystal = spga.get_refined_structure()
            c = pyxtal()
        except:
            continue
        try:
            c.from_seed(crystal, tol=0.001)
        except:
            try:
                c.from_seed(crystal, tol=0.0001)
            except:
                continue

        g = c.group.number
        num_sites = len(c.atom_sites)

        natoms = 0
        ww = []
        aa = []
        fc = []
        ws = []
        for site in c.atom_sites:
            a = element_list.index(site.specie)
            x = site.position
            m = site.wp.multiplicity
            w = letter_to_number(site.wp.letter)
            symbol = str(m) + site.wp.letter
            natoms += site.wp.multiplicity
            aa.append( a )
            ww.append( w )
            fc.append( x )  # the generator of the orbit
            ws.append( symbol )
        idx = np.argsort(ww)
        ww = np.array(ww)[idx]
        aa = np.array(aa)[idx]
        fc = np.array(fc)[idx].reshape(num_sites, 3)
        ws = np.array(ws)[idx]

        abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
        angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
        l = np.concatenate([abc, angles])

        lst.append({
            'filename':cc['filename'],
            'cif':cc['cif'],
            'n_sites':num_sites,
            'g':g,
            'l':l.tolist(),
            'fc':fc.tolist(),
            'aa':aa.tolist(),
            'ww':ww.tolist()
        })
    return lst

def save_raw_jsonl(path, out):
    cif = load_data_from_path(path)
    with open(f'{out}', 'w') as file:
        for data in cif:
            json_string = json.dumps(data)
            file.write(json_string + '\n')


def save_jsonl(path, Npool):
    if(path[-3:]=='csv'):
        cif = read_csv_file(path)
    elif(path[-3:]=='zip'):
        cif = read_zip_file(path)
    elif(path[-3:]=='.gz'):
        cif = load_data_from_tar(path)
    else:
        cif = load_data_from_path(path)

    loc = path.rfind('.')
    name = path[:loc]
    if(os.path.exists(f'{name}.jsonl')):
        return f'{name}.jsonl'

    interval = round(len(cif)/Npool)
    cif_mult = [cif[i*interval:(i+1)*interval] for i in range(Npool)]
    p = Pool(Npool)
    result = []
    for i in range(Npool):
        res = p.apply_async(process_cif, args=(cif_mult[i], i))
        result.append(res)
    p.close()
    p.join()
    new_cif=[]
    for i in range(Npool):
        temp = result[i].get()
        for j in range(len(temp)):
            new_cif.append(temp[j])

    with open(f'{name}.jsonl', 'w') as file:
        for data in new_cif:
            json_string = json.dumps(data)
            file.write(json_string + '\n')
    
    return f'{name}.jsonl'

def jsonl_to_data(path, n_max):
    G, L, XYZ, A, W = [], [], [], [], []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            num_sites = data['n_sites']
            if(n_max<num_sites):
                continue
            G.append(data['g'])
            L.append(data['l'])
            fc = np.concatenate([data['fc'], np.full((n_max - num_sites, 3), 1e10)],axis=0)
            XYZ.append(fc)
            aa = np.concatenate([data['aa'], np.full((n_max - num_sites, ), 0)],axis=0)
            A.append(aa)
            ww = np.concatenate([data['ww'], np.full((n_max - num_sites, ), 0)],axis=0)
            W.append(ww)
            
    G = np.array(G)
    L = np.array(L).reshape(-1, 6)
    XYZ = np.array(XYZ).reshape(-1, n_max, 3)
    A = np.array(A).reshape(-1, n_max)
    W = np.array(W).reshape(-1, n_max)
        
    XYZ_sort = []
    A_sort = []
    for i in range(len(A)):
        aa, xyz = sort_atoms(W[i], A[i], XYZ[i])
        A_sort.append(aa)
        XYZ_sort.append(xyz)
    A_sort = np.array(A_sort)
    XYZ_sort = np.array(XYZ_sort)
    return G, L, XYZ_sort, A_sort, W

def data_process(path, Npool, n_max):

    loc = path.rfind('.')
    name = path[:loc]
    if(os.path.exists(f'{name}_nmax_{n_max}.npz')):
        npz = np.load(f'{name}_nmax_{n_max}.npz')
        data = jnp.array(npz['array1']), jnp.array(npz['array2']), jnp.array(npz['array3']), jnp.array(npz['array4']), jnp.array(npz['array5'])
        return data

    jsonl = save_jsonl(path, Npool)
    data = jsonl_to_data(jsonl, n_max)
    
    N = len(data[0])
    random_sequence = list(range(N))
    random.shuffle(random_sequence)

    G, L, XYZ, A, W = data[0],  data[1],  data[2],  data[3],  data[4]

    loc = path.rfind('.')
    name = path[:loc]
    np.savez(f'{name}_nmax_{n_max}', 
            array1=G, array2=L, array3=XYZ, array4=A, array5=W)
    
    npz = np.load(f'{name}_nmax_{n_max}.npz')
    data = jnp.array(npz['array1']), jnp.array(npz['array2']), jnp.array(npz['array3']), jnp.array(npz['array4']), jnp.array(npz['array5'])
    return data

if __name__ == '__main__':
    argv1 = sys.argv[1]
    argv2 = int(sys.argv[2])
    _ = data_process(argv1, Npool=16, n_max=argv2)

