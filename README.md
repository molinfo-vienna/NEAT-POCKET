# NEAT-PC: <ins>P</ins>ocket-<ins>C</ins>onditioned 3D Molecular Generation with a <ins>N</ins>eighborhood-Guided, <ins>E</ins>fficient, <ins>A</ins>utoregressive Set <ins>T</ins>ransformer

Welcome to the NEAT-PC repository. NEAT is an autoregressive model that builds 3D molecules one atom at a time using a set transformer. It feeds the transformer’s output into a flow model to predict where the next atom should be by modeling the probability over its possible positions.

# Installation

1. Clone the repository and cd into the repository's root:

```bash
git clone https://github.com/molinfo-vienna/NEAT-PC.git
cd NEAT-PC
```

2. Create and activate an environment with the required python version:

```bash
conda create --name neat_pc python=3.11
conda activate neat_pc
```

3. Install PyTorch according to your hardware. For example, with GPU and CUDA 13.0 on Linux:

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

For more info, visit https://pytorch.org/get-started/locally.

4. Install NEAT-PC:

```bash
pip install -e .
```

5. Install additional PyTorch-Geometrics dependencies:

```bash
pip install pyg_lib torch_cluster torch_scatter -f https://data.pyg.org/whl/torch-2.9.0+cu130.html
```

You need to replace the last part (2.9.0+cu130) with your PyTorch version.

6. Download the model weights from TODO. Unzip and place into the project's root for using the generation script without modifications to the `config_generation_conditional.yaml` configuration file.

7. For docking score computation, add the Gnina binary to the conda environment:
```bash
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O $CONDA_PREFIX/bin/gnina
chmod +x $CONDA_PREFIX/bin/gnina
```

# Usage

## Generate molecules

0. Optionally generate BRICS fragments from either CrossDocked or SPINDR using 

```bash
python scripts/fragments_from_dataset.py --dataset <DATASET>
```

to rerun the fragment-conditioned experiments shown in the Paper.

1. Optionally change parameters in the `config_generation_conditional.yaml` file.

2. Run:

```bash
python scripts/generation_conditional.py
```

3. What you get:

- Generated molecules stored in a `generated_mols.pt` file.
- Generated molecules stored in a `generated_mols.sdf` file.
- Reference ligand `ligand.sdf`.
- Protein pocket `pocket.pdb`.


## Evaluate generated molecules

1. Optionally change parameters in the `config_evaluation.yaml` file.

2. Run:

```bash
python scripts/evaluation.py
```

3. What you get:

- Metrics per pocket, including detailled PoseBusters report per molecule.
- Average across all pockets with 95% confidence intervals.
- 2D and 3D visualizations of the first 100 generated molecules.


## Train model

1. Optionally change parameters in the `config_training.yaml` file.

2. Run:

```bash
python scripts/training.py
```

3. What you get:

- Model checkpoints (best validation loss, best validation validity and last epoch) saved in a `logs/NEAT/version_X/checkpoints` folder, along with a copy of the configuration file. The version_X folder's path should be used when loading the model's checkpoints for generating molecules or completing prefixes.


# License

This project is licensed under the MIT license.
