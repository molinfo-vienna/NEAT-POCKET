import gzip
import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import (QED, SanitizeFlags, SanitizeMol, SDMolSupplier,
                        rdFingerprintGenerator, rdMolDescriptors)
from scipy.spatial.distance import jensenshannon

plt.rcParams["font.size"] = 16

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


def bar_deviation_by_method(ax, values, methods=None, colors=None):
    """Bar plot of deviations from CrossDocked (baseline at y=0)."""
    methods = METHODS if methods is None else methods
    colors = COLORS if colors is None else colors
    for method, value, color in zip(methods, values, colors):
        ax.bar(method, value, color=color, label=method)
    ax.tick_params(axis="x", labelrotation=45)
    ax.axhline(0, color="black", linewidth=1, linestyle="-")


def safe_fraction(numerator, denominator):
    return numerator / denominator if denominator else 0.0


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


def compute_general_stats(mol):
    num_bonds = mol.GetNumBonds()
    num_atoms = mol.GetNumAtoms()
    num_heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
    frac_hetero_atoms = safe_fraction(
        rdMolDescriptors.CalcNumHeteroatoms(mol), num_atoms
    )
    frac_halogen_atoms = safe_fraction(get_num_halogen_atoms(mol), num_atoms)
    frac_rotatable_bonds = safe_fraction(
        rdMolDescriptors.CalcNumRotatableBonds(mol), num_bonds
    )
    frac_chiral_centers = safe_fraction(
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)), num_atoms
    )
    frac_hba = safe_fraction(rdMolDescriptors.CalcNumHBA(mol), num_atoms)
    frac_hbd = safe_fraction(rdMolDescriptors.CalcNumHBD(mol), num_atoms)
    logp, _ = rdMolDescriptors.CalcCrippenDescriptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    qed_score = QED.qed(mol)

    return (
        num_heavy_atoms,
        frac_hetero_atoms,
        frac_halogen_atoms,
        frac_rotatable_bonds,
        frac_chiral_centers,
        frac_hba,
        frac_hbd,
        logp,
        tpsa,
        qed_score,
    )


def compute_ring_stats(mol):
    num_atoms = mol.GetNumAtoms()
    ring_info = mol.GetRingInfo()
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    frac_aromatic_rings = safe_fraction(
        rdMolDescriptors.CalcNumAromaticRings(mol), num_rings
    )
    frac_aliphatic_rings = safe_fraction(
        rdMolDescriptors.CalcNumAliphaticRings(mol), num_rings
    )
    num_rings_3 = 0
    num_rings_4 = 0
    num_rings_5 = 0
    num_rings_6 = 0
    num_rings_7 = 0
    num_rings_8 = 0
    num_macrocycles = 0
    for x in ring_info.AtomRings():
        if len(x) == 3:
            num_rings_3 += 1
        elif len(x) == 4:
            num_rings_4 += 1
        elif len(x) == 5:
            num_rings_5 += 1
        elif len(x) == 6:
            num_rings_6 += 1
        elif len(x) == 7:
            num_rings_7 += 1
        elif len(x) == 8:
            num_rings_8 += 1
        else:
            num_macrocycles += 1
    frac_rings_3 = safe_fraction(num_rings_3, num_rings)
    frac_rings_4 = safe_fraction(num_rings_4, num_rings)
    frac_rings_5 = safe_fraction(num_rings_5, num_rings)
    frac_rings_6 = safe_fraction(num_rings_6, num_rings)
    frac_rings_7 = safe_fraction(num_rings_7, num_rings)
    frac_rings_8 = safe_fraction(num_rings_8, num_rings)
    frac_macrocycles = safe_fraction(num_macrocycles, num_rings)
    frac_bridgeheads = safe_fraction(
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol), num_atoms
    )
    frac_spiro = safe_fraction(rdMolDescriptors.CalcNumSpiroAtoms(mol), num_atoms)

    return (
        num_rings,
        frac_aromatic_rings,
        frac_aliphatic_rings,
        frac_rings_3,
        frac_rings_4,
        frac_rings_5,
        frac_rings_6,
        frac_rings_7,
        frac_rings_8,
        frac_macrocycles,
        frac_bridgeheads,
        frac_spiro,
    )


def get_mols(path):
    mols = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            if os.path.exists(os.path.join(path, dir, "generated_mols.sdf")):
                mols_file = os.path.join(path, dir, "generated_mols.sdf")
            else:
                mols_file = os.path.join(path, dir, "ligands.sdf")
            supplier = SDMolSupplier(mols_file, removeHs=False, sanitize=False)
            mols.extend([mol for mol in supplier])
    # Add aromaticity flags without sanitizing whole molecules
    [
        SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_SETAROMATICITY)
        for mol in mols
        if mol is not None
    ]
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


def js_divergence(reference, samples, bin_edges):
    """Jensen-Shannon divergence between two 1D sample distributions (base 2, in [0, 1])."""
    hist_ref, _ = np.histogram(reference, bins=bin_edges, density=False)
    hist_samp, _ = np.histogram(samples, bins=bin_edges, density=False)
    p = hist_ref / hist_ref.sum()
    q = hist_samp / hist_samp.sum()
    return float(jensenshannon(p, q, base=2) ** 2)


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

    fragment_scores_crossdocked = np.array(
        [compute_fragment_score(mol, fpscores) for mol in mols_crossdocked]
    )
    general_stats_crossdocked = np.array(
        [compute_general_stats(mol) for mol in mols_crossdocked]
    )
    ring_stats_crossdocked = np.array(
        [compute_ring_stats(mol) for mol in mols_crossdocked]
    )

    fragment_scores_pocket2mol = np.array(
        [compute_fragment_score(mol, fpscores) for mol in mols_pocket2mol]
    )
    general_stats_pocket2mol = np.array(
        [compute_general_stats(mol) for mol in mols_pocket2mol]
    )
    ring_stats_pocket2mol = np.array(
        [compute_ring_stats(mol) for mol in mols_pocket2mol]
    )

    fragment_scores_targetdiff = np.array(
        [compute_fragment_score(mol, fpscores) for mol in mols_targetdiff]
    )
    general_stats_targetdiff = np.array(
        [compute_general_stats(mol) for mol in mols_targetdiff]
    )
    ring_stats_targetdiff = np.array(
        [compute_ring_stats(mol) for mol in mols_targetdiff]
    )

    fragment_scores_diffsbdd = np.array(
        [compute_fragment_score(mol, fpscores) for mol in mols_diffsbdd]
    )
    general_stats_diffsbdd = np.array(
        [compute_general_stats(mol) for mol in mols_diffsbdd]
    )
    ring_stats_diffsbdd = np.array([compute_ring_stats(mol) for mol in mols_diffsbdd])

    fragment_scores_drugflow = np.array(
        [compute_fragment_score(mol, fpscores) for mol in mols_drugflow]
    )
    general_stats_drugflow = np.array(
        [compute_general_stats(mol) for mol in mols_drugflow]
    )
    ring_stats_drugflow = np.array([compute_ring_stats(mol) for mol in mols_drugflow])

    fragment_scores_neat = np.array(
        [compute_fragment_score(mol, fpscores) for mol in mols_neat]
    )
    general_stats_neat = np.array([compute_general_stats(mol) for mol in mols_neat])
    ring_stats_neat = np.array([compute_ring_stats(mol) for mol in mols_neat])

    ### Compute average statistics ###

    avg_fragment_scores_crossdocked = fragment_scores_crossdocked.mean(axis=0)
    avg_general_stats_crossdocked = general_stats_crossdocked.mean(axis=0)
    avg_ring_stats_crossdocked = ring_stats_crossdocked.mean(axis=0)

    avg_fragment_scores_pocket2mol = fragment_scores_pocket2mol.mean(axis=0)
    avg_general_stats_pocket2mol = general_stats_pocket2mol.mean(axis=0)
    avg_ring_stats_pocket2mol = ring_stats_pocket2mol.mean(axis=0)

    avg_fragment_scores_targetdiff = fragment_scores_targetdiff.mean(axis=0)
    avg_general_stats_targetdiff = general_stats_targetdiff.mean(axis=0)
    avg_ring_stats_targetdiff = ring_stats_targetdiff.mean(axis=0)

    avg_fragment_scores_diffsbdd = fragment_scores_diffsbdd.mean(axis=0)
    avg_general_stats_diffsbdd = general_stats_diffsbdd.mean(axis=0)
    avg_ring_stats_diffsbdd = ring_stats_diffsbdd.mean(axis=0)

    avg_fragment_scores_drugflow = fragment_scores_drugflow.mean(axis=0)
    avg_general_stats_drugflow = general_stats_drugflow.mean(axis=0)
    avg_ring_stats_drugflow = ring_stats_drugflow.mean(axis=0)

    avg_fragment_scores_neat = fragment_scores_neat.mean(axis=0)
    avg_general_stats_neat = general_stats_neat.mean(axis=0)
    avg_ring_stats_neat = ring_stats_neat.mean(axis=0)

    ### Log average statistics ###

    logging.info(f"\nFragment scores (min, mean, max):")
    logging.info(
        f"\tCrossDocked: {fragment_scores_crossdocked.min()}, {fragment_scores_crossdocked.mean()}, {fragment_scores_crossdocked.max()}"
    )
    logging.info(
        f"\tPocket2Mol: {fragment_scores_pocket2mol.min()}, {fragment_scores_pocket2mol.mean()}, {fragment_scores_pocket2mol.max()}"
    )
    logging.info(
        f"\tTargetDiff: {fragment_scores_targetdiff.min()}, {fragment_scores_targetdiff.mean()}, {fragment_scores_targetdiff.max()}"
    )
    logging.info(
        f"\tDiffSBDD: {fragment_scores_diffsbdd.min()}, {fragment_scores_diffsbdd.mean()}, {fragment_scores_diffsbdd.max()}"
    )
    logging.info(
        f"\tDrugFlow: {fragment_scores_drugflow.min()}, {fragment_scores_drugflow.mean()}, {fragment_scores_drugflow.max()}"
    )
    logging.info(
        f"\tNEAT: {fragment_scores_neat.min()}, {fragment_scores_neat.mean()}, {fragment_scores_neat.max()}"
    )

    ### Compute and log Jensen-Shannon divergence of fragment scores ###

    fragment_scores_by_method = {
        "CrossDocked": fragment_scores_crossdocked,
        "Pocket2Mol": fragment_scores_pocket2mol,
        "TargetDiff": fragment_scores_targetdiff,
        "DiffSBDD": fragment_scores_diffsbdd,
        "DrugFlow": fragment_scores_drugflow,
        "NEAT": fragment_scores_neat,
    }
    all_fragment_scores = np.concatenate(list(fragment_scores_by_method.values()))
    fragment_score_bin_edges = np.histogram_bin_edges(all_fragment_scores, bins=100)
    fragment_score_js = {
        method: js_divergence(
            fragment_scores_crossdocked,
            fragment_scores_by_method[method],
            fragment_score_bin_edges,
        )
        for method in METHODS
    }

    logging.info(
        "\nFragment score Jensen-Shannon divergence from CrossDocked "
        "(base 2, 0 = identical, 1 = maximally different):"
    )
    for method, js in sorted(fragment_score_js.items(), key=lambda x: x[1]):
        logging.info(f"\t{method}: {js:.6f}")

    ### Plot statistics ###

    # 1. General statistics

    avg_general_by_method = {
        "CrossDocked": avg_general_stats_crossdocked,
        "Pocket2Mol": avg_general_stats_pocket2mol,
        "TargetDiff": avg_general_stats_targetdiff,
        "DiffSBDD": avg_general_stats_diffsbdd,
        "DrugFlow": avg_general_stats_drugflow,
        "NEAT": avg_general_stats_neat,
    }
    general_ylabels = [
        "Δ heavy atoms",
        "Δ fraction hetero atoms",
        "Δ fraction halogen atoms",
        "Δ fraction rotatable bonds",
        "Δ fraction chiral centers",
        "Δ fraction HBA",
        "Δ fraction HBD",
        "Δ LogP",
        "Δ TPSA",
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
        "Δ fraction aromatic rings",
        "Δ fraction aliphatic rings",
        "Δ fraction bridgehead atoms",
        "Δ fraction spiro atoms",
    ]

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 18))
    for i, ylabel in enumerate(general_ylabels):
        row, col = divmod(i, 5)
        baseline = avg_general_by_method["CrossDocked"][i]
        deviations = [avg_general_by_method[method][i] - baseline for method in METHODS]
        bar_deviation_by_method(ax[row, col], deviations)
        ax[row, col].set_ylabel(ylabel)

    for i, (stat_idx, ylabel) in enumerate(zip(ring_stat_indices, ring_ylabels)):
        baseline = avg_ring_by_method["CrossDocked"][stat_idx]
        deviations = [
            avg_ring_by_method[method][stat_idx] - baseline for method in METHODS
        ]
        bar_deviation_by_method(ax[2, i], deviations)
        ax[2, i].set_ylabel(ylabel)

    ax[0, 4].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "general_stats.png")
    plt.show()

    # 2. Ring size statistics

    ring_size_labels = ["3", "4", "5", "6", "7", "8", "Macrocycle"]
    ring_size_indices = [3, 4, 5, 6, 7, 8, 9]
    ring_sizes_by_method = {
        method: {
            label: avg_stats[idx]
            for label, idx in zip(ring_size_labels, ring_size_indices)
        }
        for method, avg_stats in avg_ring_by_method.items()
    }
    ring_size_deviations_by_method = {
        method: {
            label: ring_sizes_by_method[method][label]
            - ring_sizes_by_method["CrossDocked"][label]
            for label in ring_size_labels
        }
        for method in METHODS
    }

    fig, ax = plt.subplots(figsize=(30, 12))
    grouped_bar_deviation_by_method(
        ax, ring_size_labels, ring_size_deviations_by_method
    )
    ax.set_xlabel("Ring size (number of atoms)")
    ax.set_ylabel("Δ fraction of rings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "ring_size_stats.png")
    plt.show()

    # 3. Fragment score distribution

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

    mean_pocket2mol = np.median(fragment_scores_pocket2mol)
    mean_targetdiff = np.median(fragment_scores_targetdiff)
    mean_diffsbdd = np.median(fragment_scores_diffsbdd)
    mean_drugflow = np.median(fragment_scores_drugflow)
    mean_neat = np.median(fragment_scores_neat)

    ax.axvline(
        mean_pocket2mol, color=COLOR_SCHEME["Pocket2Mol"], linestyle="--", linewidth=3
    )
    ax.axvline(
        mean_targetdiff, color=COLOR_SCHEME["TargetDiff"], linestyle="--", linewidth=3
    )
    ax.axvline(
        mean_diffsbdd, color=COLOR_SCHEME["DiffSBDD"], linestyle="--", linewidth=3
    )
    ax.axvline(
        mean_drugflow, color=COLOR_SCHEME["DrugFlow"], linestyle="--", linewidth=3
    )
    ax.axvline(mean_neat, color=COLOR_SCHEME["NEAT"], linestyle="--", linewidth=3)

    ax.set_xlabel("Fragment score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "fragment_score_distribution.png")
    plt.show()

    # 4. Atom type fractions

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
        ax[0], atom_types, atom_fractions_by_method, methods=METHODS, colors=COLORS
    )
    ax[0].set_ylabel("Fraction of atoms")
    ax[0].legend(loc="upper right")

    grouped_bar_deviation_by_method(
        ax[1], atom_types, atom_fraction_deviations_by_method
    )
    ax[1].set_xlabel("Atom type")
    ax[1].set_ylabel("Δ fraction of atoms")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "atom_fractions.png")
    plt.show()

    ### Rank methods by closeness to CrossDocked ###

    general_stat_names = [
        "heavy atoms",
        "fraction hetero atoms",
        "fraction halogen atoms",
        "fraction rotatable bonds",
        "fraction chiral centers",
        "fraction HBA",
        "fraction HBD",
        "LogP",
        "TPSA",
        "QED",
    ]
    ring_stat_names = [
        "number of rings",
        "fraction aromatic rings",
        "fraction aliphatic rings",
        "fraction 3-rings",
        "fraction 4-rings",
        "fraction 5-rings",
        "fraction 6-rings",
        "fraction 7-rings",
        "fraction 8-rings",
        "fraction macrocycles",
        "fraction bridgehead atoms",
        "fraction spiro atoms",
    ]

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
