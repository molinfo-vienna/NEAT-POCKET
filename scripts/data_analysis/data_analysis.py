import gzip
import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import (QED, SDMolSupplier, rdFingerprintGenerator,
                        rdMolDescriptors)
from scipy.spatial.distance import jensenshannon

plt.rcParams["font.size"] = 18

ROOT = Path(os.getcwd())
FPSCORES_PATH = ROOT / "scripts" / "data_analysis" / "fpscores.pkl.gz"
OUTPUT_PATH = ROOT / "output" / "data_analysis"

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


def compute_fragment_score(mol, fpscores):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sfp = mfpgen.GetSparseCountFingerprint(mol)
    frag_score = 0
    num_frag_bits = 0
    sfp_nze = sfp.GetNonzeroElements()
    for id, count in sfp_nze.items():
        num_frag_bits += count
        frag_score += fpscores.get(id, -4) * count

    return safe_fraction(frag_score, num_frag_bits)


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


def compute_js_divergence_by_method(reference_values, values_by_method):
    valid_value_sets = [values for values in values_by_method.values() if values.size]
    if not reference_values.size or not valid_value_sets:
        return {method: np.nan for method in METHODS}

    all_values = np.concatenate(valid_value_sets)
    bin_edges = np.histogram_bin_edges(all_values, bins=100)
    js_by_method = {}
    for method in METHODS:
        values = values_by_method[method]
        if not values.size:
            logging.warning(
                "Skipping fragment score Jensen-Shannon divergence for %s: no valid values",
                method,
            )
            js_by_method[method] = np.nan
            continue
        js_by_method[method] = js_divergence(reference_values, values, bin_edges)
    return js_by_method


def js_divergence(reference, samples, bin_edges):
    """Jensen-Shannon divergence between two 1D sample distributions (base 2, in [0, 1])."""
    hist_ref, _ = np.histogram(reference, bins=bin_edges, density=False)
    hist_samp, _ = np.histogram(samples, bins=bin_edges, density=False)
    p = hist_ref / hist_ref.sum()
    q = hist_samp / hist_samp.sum()
    return float(jensenshannon(p, q, base=2) ** 2)


def get_mols(path):
    mols = []
    if "crossdocked" in str(path):
        # Special loading for CrossDocked data
        mols_file = os.path.join(path, "ligands_rdkit_default.sdf")
        supplier = SDMolSupplier(mols_file, removeHs=False, sanitize=True)
        mols.extend([mol for mol in supplier])
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
                mols.extend([mol for mol in supplier])
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


def readFragmentScores(name=FPSCORES_PATH):

    data = pickle.load(gzip.open(name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    return outDict


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

    neat_path = ROOT / "output" / "version_121_cfg05_greedy" / "conditional"
    mols_neat = get_mols(neat_path)

    logging.info(f"CrossDocked: {len(mols_crossdocked)} molecules")
    logging.info(f"Pocket2Mol: {len(mols_pocket2mol)} molecules")
    logging.info(f"TargetDiff: {len(mols_targetdiff)} molecules")
    logging.info(f"DiffSBDD: {len(mols_diffsbdd)} molecules")
    logging.info(f"DrugFlow: {len(mols_drugflow)} molecules")
    logging.info(f"NEAT: {len(mols_neat)} molecules")

    ### Read fragment scores ###

    fpscores = readFragmentScores()

    ### Compute statistics ###

    fragment_scores_crossdocked = compute_property_list(
        mols_crossdocked,
        "CrossDocked",
        "fragment score",
        lambda mol: compute_fragment_score(mol, fpscores),
    )
    general_stats_crossdocked = compute_property_lists(
        mols_crossdocked, "CrossDocked", GENERAL_STAT_FUNCS
    )
    ring_stats_crossdocked = compute_property_lists(
        mols_crossdocked, "CrossDocked", RING_STAT_FUNCS
    )

    fragment_scores_pocket2mol = compute_property_list(
        mols_pocket2mol,
        "Pocket2Mol",
        "fragment score",
        lambda mol: compute_fragment_score(mol, fpscores),
    )
    general_stats_pocket2mol = compute_property_lists(
        mols_pocket2mol, "Pocket2Mol", GENERAL_STAT_FUNCS
    )
    ring_stats_pocket2mol = compute_property_lists(
        mols_pocket2mol, "Pocket2Mol", RING_STAT_FUNCS
    )

    fragment_scores_targetdiff = compute_property_list(
        mols_targetdiff,
        "TargetDiff",
        "fragment score",
        lambda mol: compute_fragment_score(mol, fpscores),
    )
    general_stats_targetdiff = compute_property_lists(
        mols_targetdiff, "TargetDiff", GENERAL_STAT_FUNCS
    )
    ring_stats_targetdiff = compute_property_lists(
        mols_targetdiff, "TargetDiff", RING_STAT_FUNCS
    )

    fragment_scores_diffsbdd = compute_property_list(
        mols_diffsbdd,
        "DiffSBDD",
        "fragment score",
        lambda mol: compute_fragment_score(mol, fpscores),
    )
    general_stats_diffsbdd = compute_property_lists(
        mols_diffsbdd, "DiffSBDD", GENERAL_STAT_FUNCS
    )
    ring_stats_diffsbdd = compute_property_lists(
        mols_diffsbdd, "DiffSBDD", RING_STAT_FUNCS
    )

    fragment_scores_drugflow = compute_property_list(
        mols_drugflow,
        "DrugFlow",
        "fragment score",
        lambda mol: compute_fragment_score(mol, fpscores),
    )
    general_stats_drugflow = compute_property_lists(
        mols_drugflow, "DrugFlow", GENERAL_STAT_FUNCS
    )
    ring_stats_drugflow = compute_property_lists(
        mols_drugflow, "DrugFlow", RING_STAT_FUNCS
    )

    fragment_scores_neat = compute_property_list(
        mols_neat,
        "NEAT",
        "fragment score",
        lambda mol: compute_fragment_score(mol, fpscores),
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

    avg_fragment_scores_crossdocked = mean_or_nan(fragment_scores_crossdocked)
    avg_general_stats_crossdocked = property_means(
        general_stats_crossdocked, general_stat_names
    )
    avg_ring_stats_crossdocked = property_means(ring_stats_crossdocked, ring_stat_names)

    avg_fragment_scores_pocket2mol = mean_or_nan(fragment_scores_pocket2mol)
    avg_general_stats_pocket2mol = property_means(
        general_stats_pocket2mol, general_stat_names
    )
    avg_ring_stats_pocket2mol = property_means(ring_stats_pocket2mol, ring_stat_names)

    avg_fragment_scores_targetdiff = mean_or_nan(fragment_scores_targetdiff)
    avg_general_stats_targetdiff = property_means(
        general_stats_targetdiff, general_stat_names
    )
    avg_ring_stats_targetdiff = property_means(ring_stats_targetdiff, ring_stat_names)

    avg_fragment_scores_diffsbdd = mean_or_nan(fragment_scores_diffsbdd)
    avg_general_stats_diffsbdd = property_means(
        general_stats_diffsbdd, general_stat_names
    )
    avg_ring_stats_diffsbdd = property_means(ring_stats_diffsbdd, ring_stat_names)

    avg_fragment_scores_drugflow = mean_or_nan(fragment_scores_drugflow)
    avg_general_stats_drugflow = property_means(
        general_stats_drugflow, general_stat_names
    )
    avg_ring_stats_drugflow = property_means(ring_stats_drugflow, ring_stat_names)

    avg_fragment_scores_neat = mean_or_nan(fragment_scores_neat)
    avg_general_stats_neat = property_means(general_stats_neat, general_stat_names)
    avg_ring_stats_neat = property_means(ring_stats_neat, ring_stat_names)

    ### Log average statistics ###

    logging.info(f"\nFragment scores (min, mean, max):")
    logging.info(
        f"\tCrossDocked: {format_distribution_summary(fragment_scores_crossdocked)}"
    )
    logging.info(
        f"\tPocket2Mol: {format_distribution_summary(fragment_scores_pocket2mol)}"
    )
    logging.info(
        f"\tTargetDiff: {format_distribution_summary(fragment_scores_targetdiff)}"
    )
    logging.info(f"\tDiffSBDD: {format_distribution_summary(fragment_scores_diffsbdd)}")
    logging.info(f"\tDrugFlow: {format_distribution_summary(fragment_scores_drugflow)}")
    logging.info(f"\tNEAT: {format_distribution_summary(fragment_scores_neat)}")

    ### Compute and log Jensen-Shannon divergence of fragment scores ###

    fragment_scores_by_method = {
        "CrossDocked": fragment_scores_crossdocked,
        "Pocket2Mol": fragment_scores_pocket2mol,
        "TargetDiff": fragment_scores_targetdiff,
        "DiffSBDD": fragment_scores_diffsbdd,
        "DrugFlow": fragment_scores_drugflow,
        "NEAT": fragment_scores_neat,
    }
    fragment_score_js = compute_js_divergence_by_method(
        fragment_scores_crossdocked, fragment_scores_by_method
    )

    logging.info(
        "\nFragment score Jensen-Shannon divergence from CrossDocked "
        "(base 2, 0 = identical, 1 = maximally different):"
    )
    for method, js in sorted(
        fragment_score_js.items(), key=lambda x: np.inf if np.isnan(x[1]) else x[1]
    ):
        logging.info(f"\t{method}: {js:.6f}")

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

    # 6. Fragment score distribution

    fig, ax = plt.subplots(figsize=(30, 12))

    ax.hist(
        fragment_scores_pocket2mol,
        bins=100,
        label="Pocket2Mol",
        histtype="step",
        density=True,
        color=COLOR_SCHEME["Pocket2Mol"],
        linewidth=2,
    )
    ax.hist(
        fragment_scores_targetdiff,
        bins=100,
        label="TargetDiff",
        histtype="step",
        density=True,
        color=COLOR_SCHEME["TargetDiff"],
        linewidth=2,
    )
    ax.hist(
        fragment_scores_diffsbdd,
        bins=100,
        label="DiffSBDD",
        histtype="step",
        density=True,
        color=COLOR_SCHEME["DiffSBDD"],
        linewidth=2,
    )
    ax.hist(
        fragment_scores_drugflow,
        bins=100,
        label="DrugFlow",
        histtype="step",
        density=True,
        color=COLOR_SCHEME["DrugFlow"],
        linewidth=2,
    )
    ax.hist(
        fragment_scores_neat,
        bins=100,
        label="NEAT",
        histtype="step",
        density=True,
        color=COLOR_SCHEME["NEAT"],
        linewidth=3,
    )

    mean_pocket2mol = safe_draw_median(
        ax, fragment_scores_pocket2mol, COLOR_SCHEME["Pocket2Mol"]
    )
    mean_targetdiff = safe_draw_median(
        ax, fragment_scores_targetdiff, COLOR_SCHEME["TargetDiff"]
    )
    mean_diffsbdd = safe_draw_median(
        ax, fragment_scores_diffsbdd, COLOR_SCHEME["DiffSBDD"]
    )
    mean_drugflow = safe_draw_median(
        ax, fragment_scores_drugflow, COLOR_SCHEME["DrugFlow"]
    )
    mean_neat = safe_draw_median(ax, fragment_scores_neat, COLOR_SCHEME["NEAT"])

    ax.set_xlabel("Fragment score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "fragment_score_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()
