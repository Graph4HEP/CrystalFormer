# CrystalFormer Exploring

## [Original README](https://github.com/deepmodeling/CrystalFormer/blob/main/README.md)

CrystalFormer is a transformer-based autoregressive model specifically designed for space group-controlled generation of crystalline materials. The space group symmetry significantly simplifies the crystal space, which is crucial for data and compute efficient generative modeling of crystalline materials. [paper](https://arxiv.org/abs/2403.15734)

## Contents

- [Contents](#contents)
- [Model card](#model-card)
- [Enviroment Setup](#enviroment-setup)
- [Available Weights](#available-weights)
- [How to run](#how-to-run)
  - [data preprocee](#data-preprocess)
  - [train](#train)
  - [parallel training](#parallel-training)
  - [sample](#sample)
  - [evaluate](#evaluate)
- [How to cite](#how-to-cite)

## Model card

The model is an autoregressive transformer for the space group conditioned crystal probability distribution `P(C|g) = P (W_1 | ... ) P ( A_1 | ... ) P(X_1| ...) P(W_2|...) ... P(L| ...)`, where

- `g`: space group number 1-230
- `W`: Wyckoff letter ('a', 'b',...,'A')
- `A`: atom type ('H', 'He', ..., 'Og')
- `X`: factional coordinates
- `L`: lattice vector [a,b,c, alpha, beta, gamma]
- `P(W_i| ...)` and `P(A_i| ...)`  are categorical distributuions.
- `P(X_i| ...)` is the mixture of von Mises distribution.
- `P(L| ...)`  is the mixture of Gaussian distribution.

We only consider symmetry inequivalent atoms. The remaining atoms are restored based on the space group and Wyckoff letter information. Note that there is a natural alphabetical ordering for the Wyckoff letters, starting with 'a' for a position with the site-symmetry group of maximal order and ending with the highest letter for the general position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). Only for the cases where discrete Wyckoff letters can not fully determine the structure, one needs to further consider factional coordinates in the loss or sampling.

## Enviroment Setup

Machine: autodl-L20, Miniconda / conda3 / python 3.10 / ubuntu 22.04 / cuda 11.8

Fork the repo, so that you can change it as you want.

Run the following command to setup the enviroment:

```bash
conda init
conda create -n crystalgpt python==3.10
source /etc/network_turbo #alternative
conda activate crystalgpt
ssh-keygen
cat ~/.ssh/id_rsa.pub #copy the public key to the ssh key setting in the github setting page
git clone git@github.com:your_name/CrystalFormer.git #clone the forked repo through ssh url, so that you can modify the code as you want
cd CrystalFormer
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple #change to a faster source of pip according to your location
pip install --upgrade "jax[cuda12]"
pip install -r requirements.txt
```

## Available Weights

The original repo release the weights of the model trained on the MP-20 dataset. More details can be seen in the their [page](https://github.com/deepmodeling/CrystalFormer/blob/main/model/README.md).

I also trained a model in an A40 machine using the same MP-20 dataset and default settings. The model and training log can be found in [here](https://drive.google.com/drive/folders/1Ip2kqEDtRp5pSwtXJn5Lk0ZoHjDzFOA_?usp=sharing).

training setting:
```bash
adam optimizer
bs: 100
lr: 0.0001
decay: 0
clip: 1
A: 119
W: 28
N: 21 
lamd a: 1
lamd w: 1
lamd l: 1
Nf: 5
Kx: 16
Kl: 4
h0: 256
layer: 16
H: 16
k: 64
m: 64
e: 32
drop: 0.5
```

## How to run

### data preprocess

[optional] the training step will auto do this, just list here to show my modification 

In the original repo, the input data is saved as csv files. The training script will read the csv files and then convert the cif strings to the strandard input format (G, L, XYZ, A, W).

Change code in 'crystalformer/src/utils.py': 
```python
def process_one(cif, atom_types, wyck_types, n_max, tol=0.01)
```
to
```python
process_one(cif, atom_types, wyck_types, n_max, tol=0.001)
```

To speed up the training if one want to re-training the model using different parameters with the same data, one can use the preprocess script to save the standard input format to a npz file.

To preprocess the data, run the following command:

```bash
python crystalformer/data/preprocess.py input_file_name max_atom_in_cell

input_file_name: name of inputs files
max_atom_in_cell: the maximum number of atoms in a cell
```

Note: 
```bash
The format of argv1 can be saved as 'tar.gz', 'zip', 'csv' or 'path contains tar.gz'.
The 'tar.gz', 'zip' should be composed of cif files. The 'csv' file should be contains a column named ['cif'].
```

Output:
```bash
the command will give 2 output files in the input path:

jsonl: contains the raw cif text and the xtal info (G, L, XYZ, A, W) extracted from pymatgen

npz: array format (G, L, XYZ, A, W) which is used to train the model
```

### train

```bash
python ./main.py --train_path data/mp_20/train.csv --valid_path data/mp_20/val.csv
```

- `train_path`: the path to the training dataset, it can be 'csv', 'tar.gz' or 'zip' files
- `valid_path`: the path to the validation dataset, it can be 'csv', 'tar.gz' or 'zip' files

### parallel training


In jax, the parallel running in a node with multi-gpus can be achieved by funciton 'pmap'. 

We add a new file `train_parallel.py` in `crystalformer/src/` to achieve the parallel training logic.

The parallel progress cannot use bool type value in the model, so the `attention.py` and `transformer.py` are also changed accordingly.

To run the parallel training, just add the '--parallel 1' option:
```bash
python main.py --parallel 1 --train_path data/mp_20/train.npz --valid_path data/mp_20/val.npz
```

### sample

```bash
python ./main.py --optimizer none --test_path data/mp_20/test.csv --restore_path model/epoch_005200.pkl --spacegroup 1 --num_samples 1000  --batchsize 1000 --temperature 1.0
```

- `optimizer`: the optimizer to use, `none` means no training, only sampling
- `restore_path`: the path to the model weights
- `spacegroup`: the space group number to sample, can be choose from 0-230, 0 means sample all labels with `num_samples`, 1-230 is sampling for a specific space group.
- `num_samples`: the number of samples to generate
- `batchsize`: the batch size for sampling
- `temperature`: the temperature for sampling

You can also use the `elements` to sample the specific element. For example, `--elements La Ni O` will sample the structure with La, Ni, and O atoms. The sampling results will be saved in the `output_LABEL.csv` file, where the `LABEL` is the space group number `g` specified in the command `--spacegroup`.

The input for the `elements` can be also the `json` file which specifies the atom mask in each Wyckoff site and the constraints. An example `atoms.json` file can be seen in the [data](./data/atoms.json) folder. There are two keys in the `atoms.json` file:

- `atom_mask`: set the atom list for each Wyckoff position, the element can only be selected from the list in the corresponding Wyckoff position
- `constraints`: set the constraints for the Wyckoff sites in the sampling, you can specify the pair of Wyckoff sites that should have the same elements

Note: 

If use parallel training, the sampling also need to add the option '--parallel 1'

### evaluate

To convert the (G, A, X, L, W) tuple to `cif` format and evaluate its structure (atoms are not too close) and compositional (charge balance) validity, run the command:
```bash
python scripts/eval.py --output_path ./model/single_gpu_result/ --label 1 
```

`output_path` is the folder which contains the pretrained model and it will also save the output results
`lable` is the space group number from 1 to 231

Calculate the novelty and uniqueness of the generated structures:

```bash
python ./scripts/compute_metrics_matbench.py --train_path TRAIN_PATH --test_path TEST_PATH --gen_path GEN_PATH --output_path OUTPUT_PATH --label SPACE_GROUP --num_io_process 40
```

- `train_path`: the path to the training dataset
- `test_path`: the path to the test dataset
- `gen_path`: the path to the generated dataset
- `output_path`: the path to save the metrics results
- `label`: the label to save the metrics results, which is the space group number `g`
- `num_io_process`: the number of processes

Note that the training, test, and generated datasets should contain the structures within the **same** space group `g` which is specified in the command `--label`.

More details about the post-processing can be seen in the [scripts](./scripts/README.md) folder.

## How to cite

If you find this repo is useful to your study, please cite the original paper

```bibtex
@misc{cao2024space,
      title={Space Group Informed Transformer for Crystalline Materials Generation}, 
      author={Zhendong Cao and Xiaoshan Luo and Jian Lv and Lei Wang},
      year={2024},
      eprint={2403.15734},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

and our study:

```bibtex
@misc{crystalformer_exploring,
  author = {Bingzhi, Li},
  title = {CrystalFormer Exploring},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {Accessed: \url{https://github.com/Graph4HEP/CrystalFormer}},
}
```

**Note**: This project is unrelated to https://github.com/omron-sinicx/crystalformer with the same name.
