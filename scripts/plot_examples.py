"""Plot example generated molecules in 5x4 grids for each model.

For each model, randomly samples 20 molecules from pockets with PoseBusters
all above MIN_PB_VALID_PERCENT (from pocket evaluation_results.txt). If fewer
than 20 eligible pockets exist, pockets are reused until 20 molecules are
collected. Saves:
  - a 3D ball-and-stick PNG (with hydrogens where applicable), oriented per
    molecule onto its first two principal components
  - a matching 2D structure PNG (explicit hydrogens removed)
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, SDMolSupplier, rdDepictor

ROOT = Path(os.getcwd())
OUTPUT_DIR = ROOT / "output" / "examples"

N_MOLECULES = 20
N_ROWS = 5
N_COLS = 4
CELL_PX = 600
DPI = 300
CELL_GAP = 0.0
VIEW_PADDING = 1.0
RANDOM_SEED = 0
MIN_PB_VALID_PERCENT = 70.0

# CPK element colors (Jmol / Mol* default palette).
CPK_COLORS = {
    1: "#D3D3D3",
    6: "#F37651",
    7: "#3050F8",
    8: "#FF0D0D",
    9: "#90E050",
    15: "#FF8000",
    16: "#FFFF30",
    17: "#1FF01F",
    35: "#A62929",
    53: "#940094",
}
DEFAULT_ATOM_COLOR = "#701F57"
STICK_LINEWIDTH = 2.5
BALL_SCALE = 0.2  # visual scaling of VdW radii for ball-and-stick
VDW_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}
DEFAULT_VDW_RADIUS = 1.70
PCA_SINGULAR_VALUE_TOL = 1e-12

@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    output_name: str
    add_hs: bool = False


MODELS: list[ModelSpec] = [
    ModelSpec(
        "Pocket2Mol",
        ROOT / "output" / "pocket2mol",
        "pocket2mol",
        add_hs=True,
    ),
    ModelSpec(
        "TargetDiff",
        ROOT / "output" / "targetdiff",
        "targetdiff",
        add_hs=True,
    ),
    ModelSpec(
        "DiffSBDD",
        ROOT / "output" / "diffsbdd",
        "diffsbdd",
        add_hs=True,
    ),
    ModelSpec(
        "DrugFlow",
        ROOT / "output" / "drugflow",
        "drugflow",
        add_hs=True,
    ),
    ModelSpec(
        "NEAT (CrossDocked)",
        ROOT / "output" / "version_126_cfg05_null",
        "neat_crossdocked",
    ),
    ModelSpec(
        "FLOWR",
        ROOT / "output" / "flowr_sample_from_train_dist",
        "flowr",
    ),
    ModelSpec(
        "NEAT (SPINDR)",
        ROOT / "output" / "version_134_cfg05_null",
        "neat_spindr",
    ),
]


def resolve_data_dir(model_path: Path) -> Path:
    conditional = model_path / "conditional"
    if conditional.is_dir():
        return conditional
    return model_path


def read_posebusters_valid_percentage(pocket_dir: Path) -> float | None:
    results_file = pocket_dir / "evaluation_results.txt"
    if not results_file.is_file():
        return None

    in_posebusters = False
    for line in results_file.read_text().splitlines():
        stripped = line.strip()
        if stripped == "PoseBusters metrics:":
            in_posebusters = True
            continue
        if not in_posebusters:
            continue
        if stripped.startswith("RDKit x PoseBusters"):
            break
        if stripped.startswith("all:"):
            return float(stripped.split(":", 1)[1].strip().rstrip("%"))
    return None


def iter_pocket_sdfs(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory not found: {root}")

    pockets: list[tuple[str, Path]] = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        sdf = subdir / "generated_mols.sdf"
        if not sdf.is_file():
            continue

        pb_valid = read_posebusters_valid_percentage(subdir)
        if pb_valid is None:
            print(f"Skipping {subdir.name}: missing PoseBusters all metric")
            continue
        if pb_valid <= MIN_PB_VALID_PERCENT:
            continue

        pockets.append((subdir.name, sdf))
    return pockets


def load_random_mol_from_pocket(
    sdf_path: Path,
    rng: random.Random,
    add_hs: bool,
) -> Chem.Mol | None:
    supplier = SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols = [
        mol
        for mol in supplier
        if mol is not None and mol.GetNumConformers() > 0
    ]
    if not mols:
        return None

    mol = rng.choice(mols)
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)
    return mol


def sample_molecules_across_pockets(
    model_path: Path,
    n: int,
    rng: random.Random,
    add_hs: bool,
) -> list[Chem.Mol]:
    root = resolve_data_dir(model_path)
    pockets = iter_pocket_sdfs(root)
    if not pockets:
        raise FileNotFoundError(
            f"No eligible pocket_*/generated_mols.sdf files under {root} "
            f"with PoseBusters all > {MIN_PB_VALID_PERCENT:.0f}%"
        )

    if len(pockets) < n:
        print(
            f"Only {len(pockets)} eligible pockets under {root}; "
            f"reusing pockets to sample {n} molecules."
        )

    selected: list[Chem.Mol] = []
    while len(selected) < n:
        added_this_round = False
        rng.shuffle(pockets)
        for _pocket_name, sdf_path in pockets:
            mol = load_random_mol_from_pocket(sdf_path, rng, add_hs=add_hs)
            if mol is None:
                print(f"Skipping {sdf_path}: no valid 3D molecules")
                continue
            selected.append(mol)
            added_this_round = True
            if len(selected) >= n:
                break
        if not added_this_round:
            break

    if len(selected) < n:
        print(
            f"Only found {len(selected)} molecules under {root} (requested {n})."
        )
    return selected[:n]


def atom_color(atom: Chem.Atom) -> str:
    return CPK_COLORS.get(atom.GetAtomicNum(), DEFAULT_ATOM_COLOR)


def atom_radius(atom: Chem.Atom) -> float:
    vdw = VDW_RADII.get(atom.GetAtomicNum(), DEFAULT_VDW_RADIUS)
    return vdw * BALL_SCALE


def bond_color(atom1: Chem.Atom, atom2: Chem.Atom) -> str:
    if atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() != 6:
        return atom_color(atom2)
    if atom2.GetAtomicNum() == 6 and atom1.GetAtomicNum() != 6:
        return atom_color(atom1)
    return atom_color(atom1)


def molecule_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    coords -= coords.mean(axis=0)
    return coords


def _complete_linear_pc2(pc1: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(pc1, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    pc2 = np.cross(pc1, helper)
    return pc2 / np.linalg.norm(pc2)


def pca_view_axes(coords: np.ndarray) -> np.ndarray:
    """Return a (2, 3) basis whose rows are PC1 and PC2 of *coords*.

    Projecting onto this plane is equivalent to viewing the molecule along
    the third principal axis (the direction of least spatial variance).
    """
    n_atoms = coords.shape[0]
    if n_atoms == 0 or not np.any(np.abs(coords) > PCA_SINGULAR_VALUE_TOL):
        return np.eye(3)[:2]

    _, singular_values, vt = np.linalg.svd(coords, full_matrices=False)
    rank = int(np.sum(singular_values > PCA_SINGULAR_VALUE_TOL))
    if rank == 0:
        return np.eye(3)[:2]

    axes = np.zeros((2, 3))
    n_keep = min(2, rank, vt.shape[0])
    axes[:n_keep] = vt[:n_keep]
    if n_keep == 1:
        axes[1] = _complete_linear_pc2(axes[0])

    projected = coords @ axes.T
    for dim in range(2):
        idx = int(np.argmax(np.abs(projected[:, dim])))
        if projected[idx, dim] < 0.0:
            axes[dim] *= -1.0
    return axes


def project_coords(coords: np.ndarray) -> np.ndarray:
    return coords @ pca_view_axes(coords).T


def mol_without_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    return Chem.RemoveHs(mol)


def prepare_mol_for_2d(mol: Chem.Mol) -> Chem.Mol | None:
    mol_no_h = mol_without_explicit_hydrogens(mol)
    try:
        rdDepictor.Compute2DCoords(mol_no_h)
    except Exception as exc:
        print(f"Warning: failed to compute 2D coordinates: {exc}")
        return None
    return mol_no_h


def compute_global_view_radius(molecules: list[Chem.Mol]) -> float:
    max_extent = 0.0
    for mol in molecules:
        xy = project_coords(molecule_coords(mol))
        radii = np.array([atom_radius(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())])
        extent_x = float(np.max(np.abs(xy[:, 0]) + radii))
        extent_y = float(np.max(np.abs(xy[:, 1]) + radii))
        max_extent = max(max_extent, extent_x, extent_y)
    return max(1e-3, max_extent * VIEW_PADDING)


def draw_sticks(ax, mol: Chem.Mol, coords: np.ndarray, view_radius: float) -> None:
    xy = project_coords(coords)
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        color = bond_color(
            mol.GetAtomWithIdx(begin_idx),
            mol.GetAtomWithIdx(end_idx),
        )
        segment = np.vstack([xy[begin_idx], xy[end_idx]])
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=color,
            linewidth=STICK_LINEWIDTH,
            solid_capstyle="round",
            antialiased=True,
            zorder=1,
        )

    for index, atom in enumerate(mol.GetAtoms()):
        radius = atom_radius(atom)
        ball = Circle(
            (xy[index, 0], xy[index, 1]),
            radius=radius,
            facecolor=atom_color(atom),
            edgecolor="none",
            antialiased=True,
            zorder=2,
        )
        ax.add_patch(ball)

    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor("white")
    ax.margins(0)


def subplot_position(index: int) -> tuple[float, float, float, float]:
    row, col = divmod(index, N_COLS)
    cell_w = 1.0 / N_COLS
    cell_h = 1.0 / N_ROWS
    width = cell_w - CELL_GAP
    height = cell_h - CELL_GAP
    left = col * cell_w + CELL_GAP / 2
    bottom = 1.0 - (row + 1) * cell_h + CELL_GAP / 2
    return left, bottom, width, height


def render_model_grid(molecules: list[Chem.Mol]) -> Image.Image:
    view_radius = compute_global_view_radius(molecules)
    cell_inches = CELL_PX / DPI
    fig = plt.figure(
        figsize=(N_COLS * cell_inches, N_ROWS * cell_inches),
        dpi=DPI,
        facecolor="white",
    )

    for index, mol in enumerate(molecules):
        left, bottom, width, height = subplot_position(index)
        ax = fig.add_axes([left, bottom, width, height])
        draw_sticks(ax, mol, molecule_coords(mol), view_radius)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        facecolor="white",
        pad_inches=0,
    )
    plt.close(fig)
    return Image.open(buffer).convert("RGB")


def render_model_grid_2d(molecules: list[Chem.Mol]) -> Image.Image | None:
    mols_2d: list[Chem.Mol] = []
    for mol in molecules:
        mol_2d = prepare_mol_for_2d(mol)
        if mol_2d is None:
            return None
        mols_2d.append(mol_2d)

    return Draw.MolsToGridImage(
        mols_2d,
        molsPerRow=N_COLS,
        subImgSize=(CELL_PX, CELL_PX),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    for model in MODELS:
        print(f"Processing {model.name} from {model.path}")
        molecules = sample_molecules_across_pockets(
            model.path,
            n=N_MOLECULES,
            rng=rng,
            add_hs=model.add_hs,
        )
        if not molecules:
            print(f"No molecules loaded for {model.name}; skipping.")
            continue

        grid_3d = render_model_grid(molecules)
        output_path_3d = OUTPUT_DIR / f"{model.output_name}.png"
        grid_3d.save(output_path_3d)
        print(f"Wrote {output_path_3d} ({len(molecules)} molecules)")

        grid_2d = render_model_grid_2d(molecules)
        if grid_2d is None:
            print(f"Skipping 2D plot for {model.name}: coordinate generation failed.")
            continue
        output_path_2d = OUTPUT_DIR / f"{model.output_name}_2d.png"
        grid_2d.save(output_path_2d)
        print(f"Wrote {output_path_2d} ({len(molecules)} molecules)")


if __name__ == "__main__":
    main()
