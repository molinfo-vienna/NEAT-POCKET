import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import QED, SDMolSupplier, rdMolDescriptors

plt.rcParams["font.size"] = 18

ROOT = Path(os.getcwd())
OUTPUT_PATH = ROOT / "output" / "data_analysis_crossdocked"

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(OUTPUT_PATH / "data_analysis.log", mode="w+")],
)
logging.info("Starting data analysis...\n")

palette = sns.color_palette("rocket")
COLOR_SCHEME = {
    "Pocket2Mol": palette[0],
    "TargetDiff": palette[1],
    "DiffSBDD": palette[2],
    "DrugFlow": palette[4],
    "NEAT": palette[5],
}  # we skip 3 because it's too similar to 2 and 4

METHODS = list(COLOR_SCHEME.keys())
COLORS = list(COLOR_SCHEME.values())


def bar_relative_by_method(ax, values, methods=None, colors=None):
    """Bar plot of deviations from CrossDocked (baseline at y=0)."""
    methods = METHODS if methods is None else methods
    colors = COLORS if colors is None else colors
    for method, value, color in zip(methods, values, colors):
        ax.bar(method, value, color=color, label=method)
    ax.tick_params(axis="x", labelrotation=45)
    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    max_abs = max(abs(v) for v in values) if values else 0.0
    if max_abs > 0:
        ax.set_ylim(-max_abs - (0.05 * max_abs), max_abs + (0.05 * max_abs))


def bar_absolute_by_method(ax, values, methods=None, colors=None):
    """Bar plot of absolute values for each method."""
    methods = METHODS if methods is None else methods
    colors = COLORS if colors is None else colors
    for method, value, color in zip(methods, values, colors):
        ax.bar(method, value, color=color, label=method)
    ax.tick_params(axis="x", labelrotation=45)


def safe_fraction(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def safe_fraction_or_nan(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def compute_atom_fractions(mols):
    counts = {}
    for mol in mols:
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
    total = sum(counts.values())
    return {symbol: safe_fraction(count, total) for symbol, count in counts.items()}


def get_ring_counts(mol):
    ring_counts = {size: 0 for size in range(3, 9)}
    num_macrocycles = 0
    for ring in mol.GetRingInfo().AtomRings():
        ring_size = len(ring)
        if ring_size in ring_counts:
            ring_counts[ring_size] += 1
        else:
            num_macrocycles += 1
    return ring_counts, num_macrocycles


GENERAL_STAT_FUNCS = [
    ("molecular weight", lambda mol: rdMolDescriptors.CalcExactMolWt(mol)),
    ("heavy atoms", lambda mol: rdMolDescriptors.CalcNumHeavyAtoms(mol)),
    (
        "fraction hetero atoms",
        lambda mol: safe_fraction(
            rdMolDescriptors.CalcNumHeteroatoms(mol), rdMolDescriptors.CalcNumAtoms(mol)
        ),
    ),
    (
        "fraction halogen atoms",
        lambda mol: safe_fraction(
            get_num_halogen_atoms(mol), rdMolDescriptors.CalcNumAtoms(mol)
        ),
    ),
    (
        "fraction rotatable bonds",
        lambda mol: safe_fraction(
            rdMolDescriptors.CalcNumRotatableBonds(mol), mol.GetNumBonds()
        ),
    ),
    (
        "fraction chiral centers",
        lambda mol: safe_fraction(
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            rdMolDescriptors.CalcNumAtoms(mol),
        ),
    ),
    (
        "fraction HBA",
        lambda mol: safe_fraction(
            rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumAtoms(mol)
        ),
    ),
    (
        "fraction HBD",
        lambda mol: safe_fraction(
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumAtoms(mol)
        ),
    ),
    ("LogP", lambda mol: rdMolDescriptors.CalcCrippenDescriptors(mol)[0]),
    ("QED", lambda mol: QED.qed(mol)),
]

RING_STAT_FUNCS = [
    ("number of rings", lambda mol: rdMolDescriptors.CalcNumRings(mol)),
    (
        "fraction aromatic rings",
        lambda mol: safe_fraction_or_nan(
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ),
    ),
    (
        "fraction aliphatic rings",
        lambda mol: safe_fraction_or_nan(
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ),
    ),
    (
        "fraction 3-rings",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[0][3], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction 4-rings",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[0][4], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction 5-rings",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[0][5], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction 6-rings",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[0][6], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction 7-rings",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[0][7], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction 8-rings",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[0][8], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction macrocycles",
        lambda mol: safe_fraction_or_nan(
            get_ring_counts(mol)[1], rdMolDescriptors.CalcNumRings(mol)
        ),
    ),
    (
        "fraction bridgehead atoms",
        lambda mol: safe_fraction(
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumAtoms(mol),
        ),
    ),
    (
        "fraction spiro atoms",
        lambda mol: safe_fraction(
            rdMolDescriptors.CalcNumSpiroAtoms(mol), rdMolDescriptors.CalcNumAtoms(mol)
        ),
    ),
]


def describe_molecule(mol, mol_idx):
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        smiles = "<SMILES unavailable>"
    return f"molecule #{mol_idx} ({smiles})"


def compute_property_list(mols, method_name, property_name, compute_fn):
    values = []
    for mol_idx, mol in enumerate(mols, start=1):
        try:
            value = compute_fn(mol)
            if np.isfinite(value):
                values.append(value)
        except Exception as exc:
            logging.warning(
                "Skipping %s for %s %s: %s",
                property_name,
                method_name,
                describe_molecule(mol, mol_idx),
                exc,
            )
    return np.array(values, dtype=float)


def compute_property_lists(mols, method_name, property_funcs):
    return {
        property_name: compute_property_list(
            mols, method_name, property_name, compute_fn
        )
        for property_name, compute_fn in property_funcs
    }


def property_means(property_lists, property_names):
    return np.array(
        [
            (
                property_lists[property_name].mean()
                if property_lists[property_name].size
                else np.nan
            )
            for property_name in property_names
        ]
    )


def log_missing_property_counts(method_name, property_lists, total_molecules):
    for property_name, values in property_lists.items():
        missing_count = total_molecules - len(values)
        if missing_count:
            logging.info(
                "\t%s: omitted %s/%s molecules for %s",
                method_name,
                missing_count,
                total_molecules,
                property_name,
            )


def format_distribution_summary(values):
    if not values.size:
        return "no valid values"
    return f"{values.min()}, {values.mean()}, {values.max()}"


def median_or_nan(values):
    return np.median(values) if values.size else np.nan


def mean_or_nan(values):
    return values.mean() if values.size else np.nan


def safe_draw_median(ax, values, color):
    median = median_or_nan(values)
    if not np.isnan(median):
        ax.axvline(median, color=color, linestyle="--", linewidth=3)
    return median


def get_mols(path):
    mols = []
    if "all_ligands" in str(path):
        # Special loading for CrossDocked data
        mols_file = os.path.join(path, "ligands_rdkit_default.sdf")
        supplier = SDMolSupplier(mols_file, removeHs=False, sanitize=True)
        mols.extend([mol for mol in supplier if mol is not None])
    else:
        # Normal loading for other data
        for dir in os.listdir(path):
            if os.path.isdir(os.path.join(path, dir)):
                if os.path.exists(os.path.join(path, dir, "generated_mols.sdf")):
                    mols_file = os.path.join(path, dir, "generated_mols.sdf")
                else:
                    logging.warning(f"No generated mols file found for {path}/{dir}")
                    continue
                supplier = SDMolSupplier(mols_file, removeHs=False, sanitize=True)
                mols.extend([mol for mol in supplier if mol is not None])
        if (
            "diffsbdd" in str(path)
            or "drugflow" in str(path)
            or "pocket2mol" in str(path)
            or "targetdiff" in str(path)
        ):
            # Add hydrogens with the RDKit default AddHs method
            mols = [Chem.AddHs(mol, addCoords=True) for mol in mols if mol is not None]
    return mols


def get_num_halogen_atoms(mol):
    num_halogen_atoms = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [9, 17, 35, 53]:
            num_halogen_atoms += 1
    return num_halogen_atoms


def grouped_bar_by_method(
    ax, categories, fractions_by_method, methods=None, colors=None
):
    methods = METHODS if methods is None else methods
    colors = COLORS if colors is None else colors
    x = np.arange(len(categories))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        offsets = x + (i - (len(methods) - 1) / 2) * width
        values = [
            fractions_by_method[method].get(category, 0) for category in categories
        ]
        ax.bar(offsets, values, width, label=method, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)


def grouped_bar_deviation_by_method(ax, categories, fractions_by_method):
    """Grouped bar plot of deviations from CrossDocked (baseline at y=0)."""
    grouped_bar_by_method(
        ax,
        categories,
        fractions_by_method,
        methods=METHODS,
        colors=COLORS,
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="-")


def ranks_by_closeness(abs_devs_by_method):
    """Rank methods by absolute deviation from CrossDocked (1 = closest)."""
    abs_devs = np.array([abs_devs_by_method[m] for m in METHODS])
    order = np.argsort(abs_devs, kind="mergesort")
    return {METHODS[idx]: rank for rank, idx in enumerate(order, start=1)}


def main() -> None:

    ### Load data ###

    crossdocked_path = ROOT / "output" / "crossdocked_all_ligands"
    mols_crossdocked = get_mols(crossdocked_path)

    pocket2mol_path = ROOT / "output" / "pocket2mol" / "conditional"
    mols_pocket2mol = get_mols(pocket2mol_path)

    targetdiff_path = ROOT / "output" / "targetdiff" / "conditional"
    mols_targetdiff = get_mols(targetdiff_path)

    diffsbdd_path = ROOT / "output" / "diffsbdd" / "conditional"
    mols_diffsbdd = get_mols(diffsbdd_path)

    drugflow_path = ROOT / "output" / "drugflow" / "conditional"
    mols_drugflow = get_mols(drugflow_path)

    neat_path = ROOT / "output" / "neat_crossdocked_cfg05_null" / "conditional"
    mols_neat = get_mols(neat_path)

    logging.info(f"CrossDocked: {len(mols_crossdocked)} molecules")
    logging.info(f"Pocket2Mol: {len(mols_pocket2mol)} molecules")
    logging.info(f"TargetDiff: {len(mols_targetdiff)} molecules")
    logging.info(f"DiffSBDD: {len(mols_diffsbdd)} molecules")
    logging.info(f"DrugFlow: {len(mols_drugflow)} molecules")
    logging.info(f"NEAT: {len(mols_neat)} molecules")

    ### Compute statistics ###

    general_stats_crossdocked = compute_property_lists(
        mols_crossdocked, "CrossDocked", GENERAL_STAT_FUNCS
    )
    ring_stats_crossdocked = compute_property_lists(
        mols_crossdocked, "CrossDocked", RING_STAT_FUNCS
    )

    general_stats_pocket2mol = compute_property_lists(
        mols_pocket2mol, "Pocket2Mol", GENERAL_STAT_FUNCS
    )
    ring_stats_pocket2mol = compute_property_lists(
        mols_pocket2mol, "Pocket2Mol", RING_STAT_FUNCS
    )

    general_stats_targetdiff = compute_property_lists(
        mols_targetdiff, "TargetDiff", GENERAL_STAT_FUNCS
    )
    ring_stats_targetdiff = compute_property_lists(
        mols_targetdiff, "TargetDiff", RING_STAT_FUNCS
    )

    general_stats_diffsbdd = compute_property_lists(
        mols_diffsbdd, "DiffSBDD", GENERAL_STAT_FUNCS
    )
    ring_stats_diffsbdd = compute_property_lists(
        mols_diffsbdd, "DiffSBDD", RING_STAT_FUNCS
    )

    general_stats_drugflow = compute_property_lists(
        mols_drugflow, "DrugFlow", GENERAL_STAT_FUNCS
    )
    ring_stats_drugflow = compute_property_lists(
        mols_drugflow, "DrugFlow", RING_STAT_FUNCS
    )

    general_stats_neat = compute_property_lists(mols_neat, "NEAT", GENERAL_STAT_FUNCS)
    ring_stats_neat = compute_property_lists(mols_neat, "NEAT", RING_STAT_FUNCS)

    logging.info("\nOmitted molecule-property pairs:")
    log_missing_property_counts(
        "CrossDocked", general_stats_crossdocked, len(mols_crossdocked)
    )
    log_missing_property_counts(
        "CrossDocked", ring_stats_crossdocked, len(mols_crossdocked)
    )
    log_missing_property_counts(
        "Pocket2Mol", general_stats_pocket2mol, len(mols_pocket2mol)
    )
    log_missing_property_counts(
        "Pocket2Mol", ring_stats_pocket2mol, len(mols_pocket2mol)
    )
    log_missing_property_counts(
        "TargetDiff", general_stats_targetdiff, len(mols_targetdiff)
    )
    log_missing_property_counts(
        "TargetDiff", ring_stats_targetdiff, len(mols_targetdiff)
    )
    log_missing_property_counts("DiffSBDD", general_stats_diffsbdd, len(mols_diffsbdd))
    log_missing_property_counts("DiffSBDD", ring_stats_diffsbdd, len(mols_diffsbdd))
    log_missing_property_counts("DrugFlow", general_stats_drugflow, len(mols_drugflow))
    log_missing_property_counts("DrugFlow", ring_stats_drugflow, len(mols_drugflow))
    log_missing_property_counts("NEAT", general_stats_neat, len(mols_neat))
    log_missing_property_counts("NEAT", ring_stats_neat, len(mols_neat))

    ### Compute average statistics ###

    general_stat_names = [name for name, _ in GENERAL_STAT_FUNCS]
    ring_stat_names = [name for name, _ in RING_STAT_FUNCS]

    avg_general_stats_crossdocked = property_means(
        general_stats_crossdocked, general_stat_names
    )
    avg_ring_stats_crossdocked = property_means(ring_stats_crossdocked, ring_stat_names)

    avg_general_stats_pocket2mol = property_means(
        general_stats_pocket2mol, general_stat_names
    )
    avg_ring_stats_pocket2mol = property_means(ring_stats_pocket2mol, ring_stat_names)

    avg_general_stats_targetdiff = property_means(
        general_stats_targetdiff, general_stat_names
    )
    avg_ring_stats_targetdiff = property_means(ring_stats_targetdiff, ring_stat_names)

    avg_general_stats_diffsbdd = property_means(
        general_stats_diffsbdd, general_stat_names
    )
    avg_ring_stats_diffsbdd = property_means(ring_stats_diffsbdd, ring_stat_names)

    avg_general_stats_drugflow = property_means(
        general_stats_drugflow, general_stat_names
    )
    avg_ring_stats_drugflow = property_means(ring_stats_drugflow, ring_stat_names)

    avg_general_stats_neat = property_means(general_stats_neat, general_stat_names)
    avg_ring_stats_neat = property_means(ring_stats_neat, ring_stat_names)

    ### Plot statistics ###

    # 1. General statistics (absolute)

    avg_general_by_method = {
        "CrossDocked": avg_general_stats_crossdocked,
        "Pocket2Mol": avg_general_stats_pocket2mol,
        "TargetDiff": avg_general_stats_targetdiff,
        "DiffSBDD": avg_general_stats_diffsbdd,
        "DrugFlow": avg_general_stats_drugflow,
        "NEAT": avg_general_stats_neat,
    }
    general_ylabels = [
        "Molecular weight",
        "Number of heavy atoms",
        "Fraction of hetero atoms",
        "Fraction of halogen atoms",
        "Fraction of rotatable bonds",
        "Fraction of chiral centers",
        "Fraction of HBA",
        "Fraction of HBD",
        "LogP",
        "QED",
    ]

    avg_ring_by_method = {
        "CrossDocked": avg_ring_stats_crossdocked,
        "Pocket2Mol": avg_ring_stats_pocket2mol,
        "TargetDiff": avg_ring_stats_targetdiff,
        "DiffSBDD": avg_ring_stats_diffsbdd,
        "DrugFlow": avg_ring_stats_drugflow,
        "NEAT": avg_ring_stats_neat,
    }
    ring_stat_indices = [0, 1, 2, 10, 11]
    ring_ylabels = [
        "Number of rings",
        "Fraction of aromatic rings",
        "Fraction of aliphatic rings",
        "Fraction of bridgehead atoms",
        "Fraction of spiro atoms",
    ]

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 18))
    for i, ylabel in enumerate(general_ylabels):
        row, col = divmod(i, 5)
        values = [avg_general_by_method[method][i] for method in METHODS]
        bar_absolute_by_method(ax[row, col], values)
        ax[row, col].set_ylabel(ylabel)

    for i, (stat_idx, ylabel) in enumerate(zip(ring_stat_indices, ring_ylabels)):
        values = [avg_ring_by_method[method][stat_idx] for method in METHODS]
        bar_absolute_by_method(ax[2, i], values)
        ax[2, i].set_ylabel(ylabel)

    ax[0, 4].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "general_stats_absolute.png")
    plt.show()

    # 2. General statistics (relative)

    avg_general_by_method = {
        "CrossDocked": avg_general_stats_crossdocked,
        "Pocket2Mol": avg_general_stats_pocket2mol,
        "TargetDiff": avg_general_stats_targetdiff,
        "DiffSBDD": avg_general_stats_diffsbdd,
        "DrugFlow": avg_general_stats_drugflow,
        "NEAT": avg_general_stats_neat,
    }
    general_ylabels = [
        "Δ molecular weight",
        "Δ number of heavy atoms",
        "Δ fraction of hetero atoms",
        "Δ fraction of halogen atoms",
        "Δ fraction of rotatable bonds",
        "Δ fraction of chiral centers",
        "Δ fraction of HBA",
        "Δ fraction of HBD",
        "Δ LogP",
        "Δ QED",
    ]

    avg_ring_by_method = {
        "CrossDocked": avg_ring_stats_crossdocked,
        "Pocket2Mol": avg_ring_stats_pocket2mol,
        "TargetDiff": avg_ring_stats_targetdiff,
        "DiffSBDD": avg_ring_stats_diffsbdd,
        "DrugFlow": avg_ring_stats_drugflow,
        "NEAT": avg_ring_stats_neat,
    }
    ring_stat_indices = [0, 1, 2, 10, 11]
    ring_ylabels = [
        "Δ number of rings",
        "Δ fraction of aromatic rings",
        "Δ fraction of aliphatic rings",
        "Δ fraction of bridgehead atoms",
        "Δ fraction of spiro atoms",
    ]

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 18))
    for i, ylabel in enumerate(general_ylabels):
        row, col = divmod(i, 5)
        baseline = avg_general_by_method["CrossDocked"][i]
        deviations = [avg_general_by_method[method][i] - baseline for method in METHODS]
        bar_relative_by_method(ax[row, col], deviations)
        ax[row, col].set_ylabel(ylabel)

    for i, (stat_idx, ylabel) in enumerate(zip(ring_stat_indices, ring_ylabels)):
        baseline = avg_ring_by_method["CrossDocked"][stat_idx]
        deviations = [
            avg_ring_by_method[method][stat_idx] - baseline for method in METHODS
        ]
        bar_relative_by_method(ax[2, i], deviations)
        ax[2, i].set_ylabel(ylabel)

    ax[0, 4].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "general_stats_relative.png")
    plt.show()

    # 3. Ring size and fraction of atoms statistics (absolute)

    ring_size_labels = ["3", "4", "5", "6", "7", "8", "Macrocycle"]
    ring_size_indices = [3, 4, 5, 6, 7, 8, 9]
    ring_sizes_by_method = {
        method: {
            label: avg_stats[idx]
            for label, idx in zip(ring_size_labels, ring_size_indices)
        }
        for method, avg_stats in avg_ring_by_method.items()
    }

    mols_by_method = {
        "CrossDocked": mols_crossdocked,
        "Pocket2Mol": mols_pocket2mol,
        "TargetDiff": mols_targetdiff,
        "DiffSBDD": mols_diffsbdd,
        "DrugFlow": mols_drugflow,
        "NEAT": mols_neat,
    }
    atom_fractions_by_method = {
        method: compute_atom_fractions(mols) for method, mols in mols_by_method.items()
    }
    periodic_table = Chem.GetPeriodicTable()
    atom_types = sorted(
        {
            symbol
            for fractions in atom_fractions_by_method.values()
            for symbol in fractions
        },
        key=periodic_table.GetAtomicNumber,
    )

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(30, 12))

    grouped_bar_by_method(
        ax[0], ring_size_labels, ring_sizes_by_method, methods=METHODS, colors=COLORS
    )
    ax[0].set_xlabel("Ring size")
    ax[0].set_ylabel("Fraction of rings")

    grouped_bar_by_method(
        ax[1], atom_types, atom_fractions_by_method, methods=METHODS, colors=COLORS
    )
    ax[1].set_xlabel("Atom type")
    ax[1].set_ylabel("Fraction of atoms")

    ax[0].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "ring_sizes_and_atom_fractions_absolute.png")
    plt.show()

    # 4. Ring size and fraction of atoms statistics (relative)

    ring_size_labels = ["3", "4", "5", "6", "7", "8", "Macrocycle"]
    ring_size_indices = [3, 4, 5, 6, 7, 8, 9]
    ring_size_deviations_by_method = {
        method: {
            label: ring_sizes_by_method[method][label]
            - ring_sizes_by_method["CrossDocked"][label]
            for label in ring_size_labels
        }
        for method in METHODS
    }

    mols_by_method = {
        "CrossDocked": mols_crossdocked,
        "Pocket2Mol": mols_pocket2mol,
        "TargetDiff": mols_targetdiff,
        "DiffSBDD": mols_diffsbdd,
        "DrugFlow": mols_drugflow,
        "NEAT": mols_neat,
    }
    atom_fractions_by_method = {
        method: compute_atom_fractions(mols) for method, mols in mols_by_method.items()
    }
    periodic_table = Chem.GetPeriodicTable()
    atom_types = sorted(
        {
            symbol
            for fractions in atom_fractions_by_method.values()
            for symbol in fractions
        },
        key=periodic_table.GetAtomicNumber,
    )

    atom_fraction_deviations_by_method = {
        method: {
            symbol: atom_fractions_by_method[method].get(symbol, 0.0)
            - atom_fractions_by_method["CrossDocked"].get(symbol, 0.0)
            for symbol in atom_types
        }
        for method in METHODS
    }

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(30, 12))

    grouped_bar_by_method(
        ax[0],
        ring_size_labels,
        ring_size_deviations_by_method,
        methods=METHODS,
        colors=COLORS,
    )
    ax[0].set_xlabel("Ring size")
    ax[0].set_ylabel("Δ fraction of rings")

    grouped_bar_by_method(
        ax[1],
        atom_types,
        atom_fraction_deviations_by_method,
        methods=METHODS,
        colors=COLORS,
    )
    ax[1].set_xlabel("Atom type")
    ax[1].set_ylabel("Δ fraction of atoms")

    ax[0].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "ring_sizes_and_atom_fractions_relative.png")
    plt.show()

    # 5. Rank methods by closeness to CrossDocked

    ranks_per_stat = {}

    for i, name in enumerate(general_stat_names):
        baseline = avg_general_by_method["CrossDocked"][i]
        abs_devs = {
            method: abs(avg_general_by_method[method][i] - baseline)
            for method in METHODS
        }
        ranks_per_stat[f"general/{name}"] = ranks_by_closeness(abs_devs)

    for i, name in enumerate(ring_stat_names):
        baseline = avg_ring_by_method["CrossDocked"][i]
        abs_devs = {
            method: abs(avg_ring_by_method[method][i] - baseline) for method in METHODS
        }
        ranks_per_stat[f"ring/{name}"] = ranks_by_closeness(abs_devs)

    for symbol in atom_types:
        abs_devs = {
            method: abs(atom_fraction_deviations_by_method[method][symbol])
            for method in METHODS
        }
        ranks_per_stat[f"atom/{symbol}"] = ranks_by_closeness(abs_devs)

    avg_ranks = {
        method: np.mean([ranks[method] for ranks in ranks_per_stat.values()])
        for method in METHODS
    }

    logging.info(
        f"\nMethod ranks by closeness to CrossDocked "
        f"(1 = closest; {len(ranks_per_stat)} statistics):"
    )
    for stat_name, ranks in ranks_per_stat.items():
        rank_str = ", ".join(f"{method}={ranks[method]}" for method in METHODS)
        logging.info(f"\t{stat_name}: {rank_str}")

    logging.info("\nAverage ranks (lower is better):")
    for method, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        logging.info(f"\t{method}: {avg_rank:.3f}")

if __name__ == "__main__":
    main()
