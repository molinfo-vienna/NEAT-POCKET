# NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer

Welcome to the NEAT-POCKET repository. NEAT is an autoregressive model that builds 3D drug-like molecules one atom at a time using a set transformer backbone. NEAT-POCKET is the protein-pocket-conditioned extension of NEAT.

# Installation

1. Clone the repository and cd into the repository's root:

```bash
git clone https://github.com/molinfo-vienna/NEAT-POCKET.git
cd NEAT-POCKET
```

2. Create and activate an environment with the required python version:

```bash
conda create --name neat-pocket python=3.11
conda activate neat-pocket
```

3. Install PyTorch according to your hardware. For example, with GPU and CUDA 13.0 on Linux:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

For more info, visit [PyTorch](https://pytorch.org/get-started/locally).

4. Install NEAT-POCKET:

```bash
pip install -e .
```

5. Install additional PyTorch-Geometrics dependencies:

```bash
pip install pyg_lib torch_cluster torch_scatter -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
```

You need to replace the last part (2.11.0+cu130) with your PyTorch version.

6. Get the model weights:

```bash
python scripts/get_weights.py
```

Alternatively you can download the trained model weights manually from [figshare](https://doi.org/10.6084/m9.figshare.33426877). Unzip and place into the project's root for using the generation script without modifications to the `config_generation_conditional.yaml` configuration file.

7. For docking score computation, add the Gnina binary to the conda environment:
```bash
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O $CONDA_PREFIX/bin/gnina
chmod +x $CONDA_PREFIX/bin/gnina
```

# Usage

## Tutorial: generate new ligands for a given protein

This tutorial shows how to generate a diverse set of ligands for a protein binding pocket. As an example, we use the AKR1C3 complex with ibuprofen from the Protein Data Bank: PDB ID 3R8G.

The goal is to generate molecules that occupy the same region of the binding pocket as ibuprofen and form similar protein–ligand interactions.

1. Input files:

We provide the necessary files in the example_input folder.

- The protein crystallographic information file: 3R8G.cif
- The ligand structure data file: IZP.sdf
- Optional: We further provide a substructure of IZP.sdf for fragment-based design (IZP_fragment.sdf)

In this example, the ligand is ibuprofen, which has the PDB ligand identifier IZP. Alternatively, you can provide your own structure data. Simply edit the protein_path and the ligand_path in the config_generation_from_pdb.yaml file accordingly. 

2. Settings

The config_generation_from_pdb.yaml file provides several options for customizing generation from a protein structure.

- Use `num_molecules` to control how many molecules are generated (default: 100).

- Use `batch_size` to pick a suitable batch size for your hardware (default: 100). E.g., we could generate a batch of 1600 molecules in parallel on an NVIDIA RTX 4090 GPU. 

- The `max_atoms` parameter defines the maximum ligand size during generation (default: 100). 

- Use `cfg_factor` to set the classifier-free-guidance scale (default: 0.5).

- Use `pocket_cutoff` to change the distance cutoff used to define the binding pocket around the ligand. We recommend keeping the default value of 6 Å because this cutoff was used during model training.

- By default, the script adds hydrogens (`add_hs`) to the protein pocket. Only disable protonation if your input protein has already been protonated using another program.

We recommend to leave all other parameters to the default setting. 


3. Run ligand generation:

```bash
python scripts/generation_from_pdb.py
```

The script automatically extracts the protein binding pocket using the same procedure as in the SPINDR dataset. By default, the pocket is defined as all protein atoms within 6 Angstrom of the reference ligand.

Note: A ligand is currently required to identify the binding pocket. However, the ligand is only used to define the pocket region and is not used during the molecule generation process. Support for apo proteins will be added in a future release.

4. Output:

After the script finishes, the output folder will contain the following files:

- pocket.pdb: the extracted protein pocket
- ligand.sdf: the hydrogenated input ligand
- generated_mols.sdf: the generated molecules

Expected runtime for 100 molecules is approximately 4 seconds on an NVIDIA GeForce RTX 4090 GPU. 

Note: The number of molecules in generated_mols.sdf may be lower than the requested number. This can happen if some generated structures are invalid or cannot be converted into RDKit molecule objects.

5. Fragment-constrained generation:

You can also generate molecules starting from a molecular fragment. To do this, provide the path to a fragment SDF in the configuration file. 


## Reproduce the results shown in the paper:

### Train model

1. Optionally change parameters in the `config_training.yaml` file.

2. Run:

```bash
python scripts/training.py
```

3. What you get:

- Model checkpoints (best validation loss, best validation validity and last epoch) saved in a `logs/NEAT/version_X/checkpoints` folder, along with a copy of the configuration file. The version_X folder's path should be used when loading the model's checkpoints for generating molecules or completing prefixes.

### Generate molecules

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


### Evaluate generated molecules

1. Optionally change parameters in the `config_evaluation.yaml` file.

2. Run:

```bash
python scripts/evaluation.py
```

3. What you get:

- Metrics per pocket, including detailled PoseBusters report per molecule.
- Average across all pockets with 95% confidence intervals.
- 2D and 3D visualizations of the first 100 generated molecules.

# Citation

If you use NEAT-POCKET, please cite it as follows:

```bibtex
@misc{jacob2026neatpocket,
      title={NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer}, 
      author={Roxane Axel Jacob and Daniel Rose and Thierry Langer and Johannes Kirchmair},
      year={2026},
      eprint={2609.05097},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.05097}, 
}
```

# License

This project is licensed under the MIT license.
