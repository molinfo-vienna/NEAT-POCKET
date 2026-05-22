"""
Taken from:
https://github.com/LPDI-EPFL/DrugFlow/blob/ed6841677ccf3590000624baa757b61ce0bb20aa/src/sbdd_metrics/metrics.py
"""

import multiprocessing
import subprocess
import tempfile
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Collection, Set, Optional
import signal
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    Descriptors,
    Crippen,
    Lipinski,
    QED,
    KekulizeException,
    AtomKekulizeException,
)


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


BOND_SYMBOLS = {
    Chem.rdchem.BondType.SINGLE: "-",
    Chem.rdchem.BondType.DOUBLE: "=",
    Chem.rdchem.BondType.TRIPLE: "#",
    Chem.rdchem.BondType.AROMATIC: ":",
}


def is_nan(value):
    return value is None or pd.isna(value) or np.isnan(value)


def safe_run(func, timeout, **kwargs):
    def _run(f, q, **kwargs):
        r = f(**kwargs)
        q.put(r)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run, kwargs={"f": func, "q": queue, **kwargs}
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        print(f"Function {func} didn't finish in {timeout} seconds. Terminating it.")
        process.terminate()
        process.join()
        return None
    elif not queue.empty():
        return queue.get()
    return None


class AbstractEvaluator:
    ID = None

    def __call__(
        self,
        molecule: Union[str, Path, Chem.Mol],
        protein: Union[str, Path] = None,
        timeout=350,
    ):
        """
        Args:
            molecule (Union[str, Path, Chem.Mol]): input molecule
            protein (str): target protein

        Returns:
            metrics (dict): dictionary of metrics
        """
        RDLogger.DisableLog("rdApp.*")
        self.check_format(molecule, protein)

        # timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(timeout)
            results = self.evaluate(molecule, protein)
        except TimeoutError:
            print(f"Error when evaluating [{self.ID}]: Timeout after {timeout} seconds")
            signal.alarm(0)
            return {}
        except Exception as e:
            print(f"Error when evaluating [{self.ID}]: {e}")
            signal.alarm(0)
            return {}
        finally:
            signal.alarm(0)
        return self.add_id(results)

    def add_id(self, results):
        if self.ID is not None:
            return {f"{self.ID}.{key}": value for key, value in results.items()}
        else:
            return results

    @abstractmethod
    def evaluate(
        self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path]
    ) -> Dict[str, Union[int, float, str]]:
        raise NotImplementedError

    @staticmethod
    def check_format(molecule, protein):
        assert isinstance(
            molecule, (str, Path, Chem.Mol)
        ), "Supported molecule types: str, Path, Chem.Mol"
        assert protein is None or isinstance(
            protein, (str, Path)
        ), "Supported protein types: str"
        if isinstance(molecule, (str, Path)):
            supp = Chem.SDMolSupplier(str(molecule), sanitize=False)
            assert len(supp) == 1, "Only one molecule per file is supported"

    @staticmethod
    def load_molecule(molecule):
        if isinstance(molecule, (str, Path)):
            return Chem.SDMolSupplier(str(molecule), sanitize=False)[0]
        return Chem.Mol(
            molecule
        )  # create copy to avoid overriding properties of the input molecule

    @staticmethod
    def save_molecule(molecule, sdf_path):
        if isinstance(molecule, (str, Path)):
            return molecule

        with Chem.SDWriter(str(sdf_path)) as w:
            try:
                w.write(molecule)
            except (RuntimeError, ValueError) as e:
                if isinstance(e, (KekulizeException, AtomKekulizeException)):
                    w.SetKekulize(False)
                    w.write(molecule)
                    w.SetKekulize(True)
                else:
                    w.write(Chem.Mol())
                    print("[AbstractEvaluator] Error when saving the molecule")

        return sdf_path

    @property
    def dtypes(self):
        return self.add_id(self._dtypes)

    @property
    @abstractmethod
    def _dtypes(self):
        raise NotImplementedError


class GninaEvalulator(AbstractEvaluator):
    ID = "gnina"

    def __init__(self, gnina="gnina"):
        self.gnina = gnina

    def evaluate(self, molecule, protein=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            molecule = self.save_molecule(
                molecule, sdf_path=Path(tmpdir, "molecule.sdf")
            )
            gnina_cmd = f"{self.gnina} -r {str(protein)} -l {str(molecule)} --minimize --seed 42 --no_gpu"
            gnina_result = subprocess.run(
                gnina_cmd, shell=True, capture_output=True, text=True
            )
            n_atoms = self.load_molecule(molecule).GetNumAtoms()

        gnina_scores = self.read_gnina_results(gnina_result)

        # Additionally computing ligand efficiency
        gnina_scores["vina_efficiency"] = (
            gnina_scores["vina_score"] / n_atoms if n_atoms > 0 else None
        )
        gnina_scores["gnina_efficiency"] = (
            gnina_scores["gnina_score"] / n_atoms if n_atoms > 0 else None
        )
        return gnina_scores

    @staticmethod
    def read_gnina_results(gnina_result):
        res = {
            "vina_score": None,
            "gnina_score": None,
            "minimisation_rmsd": None,
            "cnn_score": None,
        }
        if gnina_result.returncode != 0:
            print(gnina_result.stderr)
            return res

        for line in gnina_result.stdout.split("\n"):
            if line.startswith("Affinity"):
                res["vina_score"] = float(line.split(" ")[1].strip())
            if line.startswith("CNNaffinity"):
                res["gnina_score"] = float(line.split(" ")[1].strip())
            if line.startswith("CNNscore"):
                res["cnn_score"] = float(line.split(" ")[1].strip())
            if line.startswith("RMSD"):
                res["minimisation_rmsd"] = float(line.split(" ")[1].strip())

        return res

    @property
    def _dtypes(self):
        return {"*": float}


class ClashEvaluator(AbstractEvaluator):
    ID = "clashes"

    def __init__(self, margin=0.75, ignore={"H"}):
        self.margin = margin
        self.ignore = ignore

    def evaluate(self, molecule=None, protein=None):
        result = {
            "passed_clash_score_ligands": None,
            "passed_clash_score_pockets": None,
            "passed_clash_score_between": None,
        }
        if molecule is not None:
            molecule = self.load_molecule(molecule)
            clash_score = self.clash_score(molecule)
            result["clash_score_ligands"] = clash_score
            result["passed_clash_score_ligands"] = clash_score == 0

        if protein is not None:
            protein = Chem.MolFromPDBFile(str(protein), sanitize=False)
            clash_score = self.clash_score(protein)
            result["clash_score_pockets"] = clash_score
            result["passed_clash_score_pockets"] = clash_score == 0

        if molecule is not None and protein is not None:
            clash_score = self.clash_score(molecule, protein)
            result["clash_score_between"] = clash_score
            result["passed_clash_score_between"] = clash_score == 0

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
        return np.mean(clashes)

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

    @property
    def _dtypes(self):
        return {
            "clash_score_ligands": float,
            "clash_score_pockets": float,
            "clash_score_between": float,
            "passed_clash_score_ligands": bool,
            "passed_clash_score_pockets": bool,
            "passed_clash_score_between": bool,
        }
