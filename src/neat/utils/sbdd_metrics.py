"""
Taken from:
https://github.com/LPDI-EPFL/DrugFlow/blob/ed6841677ccf3590000624baa757b61ce0bb20aa/src/sbdd_metrics/metrics.py
"""

import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
from rdkit import Chem

from .utils import center_pdb


class GninaEvaluator:
    ID = "gnina"

    def __init__(self, gnina="gnina"):
        self.gnina = gnina

    def evaluate_mols(
        self,
        sdf_path,
        pdb_path=None,
        ligand_path=None,
        minimize=False,
        cnn_scoring=False,
    ) -> dict[str, float]:
        if minimize:
            minimize_flag = "--minimize"
        else:
            minimize_flag = "--score_only"
        if not cnn_scoring:
            minimize_flag += " --cnn_scoring none"

        # gnina result of the original ligand
        if ligand_path is not None:
            gnina_cmd = f"{self.gnina} -r {str(pdb_path)} -l {str(ligand_path)} {minimize_flag} --seed 42 --no_gpu"
            gnina_result = subprocess.run(
                gnina_cmd, shell=True, capture_output=True, text=True
            )
            ligand_scores = self.read_gnina_results(
                gnina_result, cnn_scoring=cnn_scoring
            )

        # gnina results of the generated mols
        gnina_cmd = f"{self.gnina} -r {str(pdb_path)} -l {str(sdf_path)} {minimize_flag} --seed 42 --no_gpu"
        gnina_result = subprocess.run(
            gnina_cmd, shell=True, capture_output=True, text=True
        )
        generated_mol_scores = self.read_gnina_results(
            gnina_result, cnn_scoring=cnn_scoring
        )
        if ligand_path is not None:
            generated_mol_scores["high_affinity"] = list(
                np.array(generated_mol_scores["vina_score"])
                < ligand_scores["vina_score"][0]
            )
        mean_results = {
            key: np.array(values).mean() if values else 0.0
            for key, values in generated_mol_scores.items()
        }

        return mean_results

    @staticmethod
    def read_gnina_results(gnina_result, cnn_scoring=False):

        if cnn_scoring:
            metrics = {
                "vina_score": [],
                "gnina_score": [],
                "cnn_score": [],
                "minimisation_rmsd": [],
            }
        else:
            metrics = {
                "vina_score": [],
            }

        if gnina_result.returncode != 0:
            print(gnina_result.stderr)
            return metrics

        # Step 1: Parse and collect all data points
        for line in gnina_result.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Split by whitespace, ignoring multiple spaces

            if line.startswith("Affinity:"):
                try:
                    value = float(line.split()[1])
                    metrics["vina_score"].append(value)
                except Exception as e:
                    print(f"Error parsing line: '{line}'. Error: {e}")
                    continue

            elif line.startswith("CNNaffinity:") and cnn_scoring:
                try:
                    value = float(line.split()[1])
                    metrics["gnina_score"].append(value)
                except Exception as e:
                    print(f"Error parsing line: '{line}'. Error: {e}")
                    continue

            elif line.startswith("CNNscore:") and cnn_scoring:
                try:
                    value = float(line.split()[1])
                    metrics["cnn_score"].append(value)
                except Exception as e:
                    print(f"Error parsing line: '{line}'. Error: {e}")
                    continue

            elif line.startswith("RMSD:") and cnn_scoring:
                try:
                    value = float(line.split()[1])
                    metrics["minimisation_rmsd"].append(value)
                except Exception as e:
                    print(f"Error parsing line: '{line}'. Error: {e}")
                    continue

        return metrics


class ClashEvaluator:
    ID = "clashes"

    def __init__(self, margin=0.75, ignore={"H"}):
        self.margin = margin
        self.ignore = ignore

    def evaluate_mol(self, molecule=None, protein=None):
        result = {}
        protein = Chem.MolFromPDBFile(str(protein), sanitize=False)

        clash_score_mean, clash_score_sum = self.clash_score(molecule, protein)
        result["clash_score_between_mean"] = clash_score_mean
        result["clash_score_between_sum"] = clash_score_sum
        result["passed_clash_score_between"] = clash_score_mean == 0

        return result

    def clash_score(self, rdmol1, rdmol2=None):
        """
        Computes a clash score as the number of atoms that have at least one
        clash divided by the number of atoms in the molecule.

        INTERMOLECULAR CLASH SCORE
        If rdmol2 is provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with rdmol2.
        We define a clash if two atoms are closer than "margin times the sum of
        their van der Waals radii".

        INTRAMOLECULAR CLASH SCORE
        If rdmol2 is not provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with other atoms in rdmol1.
        In this case, a clash is defined by margin times the atoms' smallest
        covalent radii (among single, double and triple bond radii). This is done
        so that this function is applicable even if no connectivity information is
        available.
        """

        intramolecular = rdmol2 is None
        if intramolecular:
            rdmol2 = rdmol1

        coord1, radii1 = self.coord_and_radii(rdmol1, intramolecular=intramolecular)
        coord2, radii2 = self.coord_and_radii(rdmol2, intramolecular=intramolecular)

        dist = np.sqrt(np.sum((coord1[:, None, :] - coord2[None, :, :]) ** 2, axis=-1))
        if intramolecular:
            np.fill_diagonal(dist, np.inf)

        clashes = dist < self.margin * (radii1[:, None] + radii2[None, :])
        clashes = np.any(clashes, axis=1)
        return np.mean(clashes), np.sum(clashes)

    def coord_and_radii(self, rdmol, intramolecular):
        _periodic_table = Chem.GetPeriodicTable()
        _get_radius = (
            _periodic_table.GetRcovalent if intramolecular else _periodic_table.GetRvdw
        )

        coord = rdmol.GetConformer().GetPositions()
        radii = np.array([_get_radius(a.GetSymbol()) for a in rdmol.GetAtoms()])

        mask = np.array([a.GetSymbol() not in self.ignore for a in rdmol.GetAtoms()])
        coord = coord[mask]
        radii = radii[mask]

        assert coord.shape[0] == radii.shape[0]
        return coord, radii

    def evaluate_mols(self, mols, pocket_path) -> dict[str, float]:
        with NamedTemporaryFile(delete=True, suffix=".pdb") as temp_file:
            # Convert the string path to a pathlib.Path object
            center_pdb(pocket_path, temp_file.name)
            scores_mean = []
            scores_sum = []
            no_clashes = []

            for mol in mols:
                if mol is None:
                    continue
                clash_results = self.evaluate_mol(mol, temp_file.name)
                clash_score_mean = clash_results["clash_score_between_mean"]
                clash_score_sum = clash_results["clash_score_between_sum"]
                no_clashes.append(clash_results["passed_clash_score_between"])
                scores_mean.append(clash_score_mean)
                scores_sum.append(clash_score_sum)

            score_mean = np.array(scores_mean).mean() if len(scores_mean) > 0 else 0.0
            score_sum = np.array(scores_sum).mean() if len(scores_sum) > 0 else 0.0
            no_clashes = np.array(no_clashes).mean() if len(no_clashes) > 0 else 0.0

            return {
                "clash_score_mean": float(score_mean),
                "clash_score_sum": float(score_sum),
                "no_clashes": float(no_clashes),
            }
