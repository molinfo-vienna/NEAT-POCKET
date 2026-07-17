import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from neat.model.molecule_builder import MoleculeBuilder

plt.rcParams["font.size"] = 12

ROOT = Path(os.getcwd())
OUTPUT_PATH = ROOT / "output" / "data_analysis"

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def setup_looger(name, filename, level=logging.INFO):
    if os.path.exists(filename):
        os.remove(filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

types_mapping = {
    1: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    17: 11,
    33: 12,
    35: 13,
    53: 14,
    80: 15,
    83: 16,
}

symbols_mapping = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
}

masses_mapping = {
    1: 1.008,
    5: 10.811,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    15: 30.974,
    16: 32.065,
    17: 35.453,
    35: 79.904,
}

def compute_atom_ratios(atom_types):
    atom_ratios = {}
    for element in atom_types:
        if element not in atom_ratios:
            atom_ratios[element] = 0
        atom_ratios[element] += 1
    return {k: v / len(atom_types) for k, v in atom_ratios.items()}


def main() -> None:

    logging_filename = os.path.join(os.getcwd(), "output", "data_analysis", "raw_molecules_analysis.log")
    logger = setup_looger("raw_molecules_analysis", logging_filename)
    logger.info("Starting raw molecules analysis...\n")
    
    # Load data
    neat_path = ROOT / "output" / "version_121_cfg05_greedy" / "conditional"

    x_list = []
    pos_list = []
    batch_list = []

    for subdir in neat_path.iterdir():
        if not subdir.is_dir():
            continue
        if not (subdir / "generated_mols.pt").exists():
            continue
        builder = MoleculeBuilder(vocab="CROSSDOCKED")
        subx, subpos, subbatch = builder.load_tensor_from_file(subdir)
        x_list.append(subx)
        pos_list.append(subpos)
        batch_list.append(subbatch)
    
    x = torch.cat(x_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    batch = torch.cat(batch_list, dim=0)

    # Reindex batch so that it is consecutive across all pockets
    # e.g. 0,0,0,1,1,2,2,2,0,0,0,1,1,1 -> 0,0,0,1,1,2,2,2,3,3,3,4,4,4
    new_indices = []
    current_pocket = batch[0].item()
    current_index = 0
    for value in batch:
        if value.item() != current_pocket:
            current_pocket = value.item()
            current_index += 1
        new_indices.append(current_index)
    batch_reindexed = torch.tensor(new_indices)

    inv_types_mapping = {v: k for k, v in types_mapping.items()}
    atom_types = [inv_types_mapping[element.item()] for element in x]
    atom_symbols = [symbols_mapping[element] for element in atom_types]
    atom_masses = [masses_mapping[element] for element in atom_types]
    atom_symbols_ratios = compute_atom_ratios(atom_symbols)
    logger.info(f"Atom symbols ratios: {atom_symbols_ratios}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(atom_symbols_ratios.keys(), atom_symbols_ratios.values())
    ax.set_xlabel("Atom")
    ax.set_ylabel("Ratio")
    ax.set_title("Atom ratios")
    plt.savefig(OUTPUT_PATH / "raw_atom_types_fractions.png")

    atom_masses_tensor = torch.tensor(atom_masses)
    molecular_masses = torch.zeros(batch_reindexed.max().item() + 1, dtype=torch.float32)
    molecular_masses = torch.scatter_add(molecular_masses, 0, batch_reindexed, atom_masses_tensor)

    molecular_masses_min, molecular_masses_mean, molecular_masses_max = torch.min(molecular_masses).item(), torch.mean(molecular_masses).item(), torch.max(molecular_masses).item()
    logger.info(f"Molecular masses min: {molecular_masses_min}, molecular masses mean: {molecular_masses_mean}, molecular masses max: {molecular_masses_max}")

if __name__ == "__main__":
    main()
