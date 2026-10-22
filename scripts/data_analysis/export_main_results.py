"""
This script is used to export the data behind the main result tables of the NEAT-POCKET paper:
    1. Performance of NEAT-POCKET vs. other pocket-conditioned generators (CrossDocked and SPINDR)
    2. Impact of CFG on NEAT-POCKET trained on CrossDocked
    3. Impact of CFG on NEAT-POCKET trained on SPINDR
    4. Fragment (prefix) completion of NEAT-POCKET vs. DiffSBDD (CrossDocked)
    5. Fragment (prefix) completion of NEAT-POCKET vs. FLOWR (SPINDR)

All metrics are recomputed from the per-pocket evaluation_results.txt files written by
scripts/evaluation.py under generated_data/<run>/conditional/pocket_*/. For each run, the
per-pocket values are aggregated across pockets as follows:
    - PC SE (PoseCheck strain energy): median ± median absolute deviation (MAD), because the
      per-pocket strain energies are highly skewed for some models (see investigate_strain_energies.py)
    - all other metrics: mean ± half-width of the 95% Student-t confidence interval, as in
      evaluation.py

Two corrections are applied when aggregating:
    - Pockets with a generated_mols.sdf but no evaluation_results.txt (no molecule could be
      constructed, so evaluation.py skipped them) count as 0% for RDKit valid and PB valid and are
      excluded from all other metrics.
    - evaluation.py pads every pocket to 100 molecules, so the RDKit valid and PB valid fractions of
      the test sets (one reference ligand per pocket) are rescaled by 100.

The Rank column is the average rank of each model by closeness to the training-set distribution of
molecular properties. It is read from the logs written by data_analysis_crossdocked.py and
data_analysis_spindr.py, which must be run first. Sampling times are not stored in generated_data/
and are therefore not exported. These were obtained manually by running each model.

It produces the following outputs in the output/data_analysis/main_results directory:
    1. main_results.log: the five tables
    2. main_results.csv: unrounded value, error and number of pockets for every run and metric

The script must be run from the repository root.
"""

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(os.getcwd())
INPUT_PATH = ROOT / "generated_data"
DATA_ANALYSIS_PATH = ROOT / "output" / "data_analysis"
OUTPUT_PATH = DATA_ANALYSIS_PATH / "main_results"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_PATH / "main_results.log", mode="w+"),
        logging.StreamHandler(),
    ],
)

LINE_RE = re.compile(r"^(?P<name>[^:]+):\s*(?P<value>-?(?:\d+(?:\.\d+)?|nan))%?\s*$")
RANK_RE = re.compile(r"^\s*(?P<name>\S+):\s*(?P<value>-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class Metric:
    header: str
    key: str  # "<section>/<name>" in evaluation_results.txt, or "rank"
    decimals: int
    aggregation: str = "mean_ci"  # "mean_ci" or "median_mad"
    zero_if_missing: bool = False
    rescale_reference: bool = False


RDKIT_VALID = Metric("RDKit valid [%]", "RDKit metrics/Valid", 1,
                     zero_if_missing=True, rescale_reference=True)
PB_VALID = Metric("PB valid [%]", "RDKit x PoseBusters", 1,
                  zero_if_missing=True, rescale_reference=True)
PC_CLASHES = Metric("PC clashes [1]", "PoseCheck metrics/clashes", 1)
PC_SE = Metric("PC SE [kcal/mol]", "PoseCheck metrics/strain", 0, aggregation="median_mad")
VINA_SCORE = Metric("Vina score [kcal/mol]", "Vina metrics/vina_score", 1)
VINA_MIN = Metric("Vina min [kcal/mol]", "Vina metrics/vina_score_min", 1)
MOL_WEIGHT = Metric("Mol. weight [g/mol]", "Physchem metrics/MW", 0)
UNIQUE = Metric("Unique [%]", "RDKit metrics/Unique", 1)
RANK = Metric("Rank [1]", "rank", 2)
EXPORTED_METRICS = [RDKIT_VALID, PB_VALID, PC_CLASHES, PC_SE, VINA_SCORE, VINA_MIN, MOL_WEIGHT, UNIQUE]


@dataclass
class Row:
    label: str
    run: str  # directory under generated_data/
    rank_name: str | None = None  # "<dataset>/<method>" as in the data analysis logs
    is_reference: bool = False


@dataclass
class Group:
    title: str | None
    rows: list[Row]


@dataclass
class Table:
    title: str
    metrics: list[Metric]
    groups: list[Group] = field(default_factory=list)


FRAGMENT_TYPES = ["smallest", "second_largest", "largest"]

TABLES = [
    Table(
        "Table 1: Performance of NEAT-POCKET vs. other pocket-conditioned 3D molecule generators",
        [RDKIT_VALID, PB_VALID, PC_CLASHES, PC_SE, VINA_SCORE, VINA_MIN, MOL_WEIGHT, RANK],
        [
            Group("CrossDocked", [
                Row("CrossDocked", "crossdocked_test", is_reference=True),
                Row("Pocket2Mol", "pocket2mol", "crossdocked/Pocket2Mol"),
                Row("TargetDiff", "targetdiff", "crossdocked/TargetDiff"),
                Row("DiffSBDD", "diffsbdd", "crossdocked/DiffSBDD"),
                Row("DrugFlow", "drugflow", "crossdocked/DrugFlow"),
                Row("NEAT-POCKET", "neat_crossdocked_cfg05_null", "crossdocked/NEAT"),
            ]),
            Group("SPINDR", [
                Row("SPINDR", "spindr_test", is_reference=True),
                Row("FLOWR", "flowr", "spindr/FLOWR"),
                Row("NEAT-POCKET", "neat_spindr_cfg05_null", "spindr/NEAT"),
            ]),
        ],
    ),
    Table(
        "Table 2: Impact of CFG on NEAT-POCKET trained on CrossDocked",
        [PB_VALID, PC_CLASHES, PC_SE, VINA_SCORE, VINA_MIN, MOL_WEIGHT],
        [Group(None, [
            Row(f"{cfg / 10:.1f}", f"neat_crossdocked_cfg{cfg:02d}_null") for cfg in range(0, 30, 5)
        ])],
    ),
    Table(
        "Table 3: Impact of CFG on NEAT-POCKET trained on SPINDR",
        [PB_VALID, PC_CLASHES, PC_SE, VINA_SCORE, VINA_MIN, MOL_WEIGHT],
        [Group(None, [
            Row(f"{cfg / 10:.1f}", f"neat_spindr_cfg{cfg:02d}_null") for cfg in range(0, 30, 5)
        ])],
    ),
    Table(
        "Table 4: Fragment completion of NEAT-POCKET vs. DiffSBDD (CrossDocked)",
        [RDKIT_VALID, PB_VALID, PC_CLASHES, PC_SE, VINA_SCORE, VINA_MIN, MOL_WEIGHT, UNIQUE],
        [
            Group("DiffSBDD", [Row("none", "diffsbdd")] + [
                Row(f.replace("_", "-"), f"diffsbdd_fragments/{f}_filtered") for f in FRAGMENT_TYPES
            ]),
            Group("NEAT-POCKET (CrossDocked)", [Row("none", "neat_crossdocked_cfg05_null")] + [
                Row(f.replace("_", "-"), f"neat_crossdocked_cfg05_{f}") for f in FRAGMENT_TYPES
            ]),
        ],
    ),
    Table(
        "Table 5: Fragment completion of NEAT-POCKET vs. FLOWR (SPINDR)",
        [RDKIT_VALID, PB_VALID, PC_CLASHES, PC_SE, VINA_SCORE, VINA_MIN, MOL_WEIGHT, UNIQUE],
        [
            Group("FLOWR", [Row("none", "flowr")] + [
                Row(f.replace("_", "-"), f"flowr_fragments/{f}_filtered") for f in FRAGMENT_TYPES
            ]),
            Group("NEAT-POCKET (SPINDR)", [Row("none", "neat_spindr_cfg05_null")] + [
                Row(f.replace("_", "-"), f"neat_spindr_cfg05_{f}") for f in FRAGMENT_TYPES
            ]),
        ],
    ),
]


def parse_evaluation_results(path: Path) -> dict[str, float]:
    """Parse evaluation_results.txt into {"<section>/<name>": value}; percentages stay in %."""
    metrics = {}
    section = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.endswith(":"):
            section = line[:-1]
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip()
        key = name if name == "RDKit x PoseBusters" else f"{section}/{name}"
        metrics[key] = float(match.group("value"))
    return metrics


def load_run(run: str) -> list[dict[str, float] | None]:
    """Per-pocket metrics of a run; None for pockets with generated molecules but no results."""
    data_dir = INPUT_PATH / run / "conditional"
    pockets = sorted(p for p in data_dir.glob("pocket_*") if (p / "generated_mols.sdf").exists())
    if not pockets:
        raise FileNotFoundError(f"No pockets with generated_mols.sdf under {data_dir}")
    results = []
    for pocket in pockets:
        path = pocket / "evaluation_results.txt"
        results.append(parse_evaluation_results(path) if path.exists() else None)
    return results


def mean_and_95_ci(values: np.ndarray) -> tuple[float, float]:
    """Same as compute_mean_and_95_ci in evaluation.py."""
    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0
    std_err = np.std(values, ddof=1) / np.sqrt(n)
    return float(np.mean(values)), float(stats.t.ppf(0.975, df=n - 1) * std_err)


def median_and_mad(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    return median, float(np.median(np.abs(values - median)))


def aggregate(pockets: list[dict | None], metric: Metric, is_reference: bool) -> tuple[float, float, int]:
    values = []
    for pocket in pockets:
        if pocket is None:
            if metric.zero_if_missing:
                values.append(0.0)
            continue
        value = pocket.get(metric.key, np.nan)
        if not np.isnan(value):
            values.append(value)
    values = np.asarray(values, dtype=float)
    if metric.rescale_reference and is_reference:
        values = values * 100
    if len(values) == 0:
        return np.nan, np.nan, 0
    if metric.aggregation == "median_mad":
        return (*median_and_mad(values), len(values))
    return (*mean_and_95_ci(values), len(values))


def load_ranks() -> dict[str, float]:
    """Average ranks from the data analysis logs, keyed by "<dataset>/<method>"."""
    ranks = {}
    for dataset in ["crossdocked", "spindr"]:
        log_path = DATA_ANALYSIS_PATH / dataset / "data_analysis.log"
        if not log_path.exists():
            raise FileNotFoundError(
                f"{log_path} not found; run scripts/data_analysis/data_analysis_{dataset}.py first"
            )
        lines = log_path.read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("Average ranks"))
        for line in lines[start + 1:]:
            match = RANK_RE.match(line)
            if not match:
                break
            ranks[f"{dataset}/{match.group('name')}"] = float(match.group("value"))
    return ranks


def round_half_up(value: float, decimals: int) -> float:
    """Round exact halves away from zero (e.g. 71.05 -> 71.1), unlike Python's round()."""
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(round(value, 8))).quantize(quantum, rounding=ROUND_HALF_UP))


def format_cell(value: float, error: float | None, decimals: int) -> str:
    if np.isnan(value):
        return "-"
    text = f"{round_half_up(value, decimals):.{decimals}f}"
    if error is None:
        return text
    return f"{text} ± {round_half_up(error, decimals):.{decimals}f}"


def log_table(table: Table, results: dict, ranks: dict) -> None:
    header = ["", *[m.header for m in table.metrics]]
    lines = []
    for group in table.groups:
        cells = {id(row): [row.label] for row in group.rows}
        for metric in table.metrics:
            for row in group.rows:
                if metric is RANK:
                    value = ranks.get(row.rank_name, np.nan) if row.rank_name else np.nan
                    error = None
                else:
                    value, error = results[row.run][metric.key][:2]
                cells[id(row)].append(format_cell(value, error, metric.decimals))
        lines.append(("group", group.title))
        for row in group.rows:
            lines.append(("row", cells[id(row)]))

    widths = [len(h) for h in header]
    for kind, content in lines:
        if kind == "row":
            widths = [max(w, len(c)) for w, c in zip(widths, content)]
    separator = "-" * (sum(widths) + 3 * (len(widths) - 1))

    logging.info(f"\n{table.title}")
    logging.info(separator)
    logging.info(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    for kind, content in lines:
        if kind == "group":
            logging.info(separator)
            if content:
                logging.info(content)
        else:
            logging.info(" | ".join(c.rjust(w) if i else c.ljust(w)
                                    for i, (c, w) in enumerate(zip(content, widths))))
    logging.info(separator)


def main() -> None:
    reference_runs = {row.run for t in TABLES for g in t.groups for row in g.rows if row.is_reference}
    runs = list(dict.fromkeys(row.run for t in TABLES for g in t.groups for row in g.rows))

    results = {}
    csv_rows = []
    for run in runs:
        pockets = load_run(run)
        n_missing = sum(p is None for p in pockets)
        if n_missing:
            logging.info(
                f"{run}: {n_missing}/{len(pockets)} pocket(s) without evaluation results "
                "(counted as 0% RDKit valid and PB valid)"
            )
        results[run] = {}
        for metric in EXPORTED_METRICS:
            value, error, n = aggregate(pockets, metric, run in reference_runs)
            results[run][metric.key] = (value, error, n)
            csv_rows.append({
                "run": run,
                "metric": metric.header,
                "aggregation": "median ± MAD" if metric.aggregation == "median_mad" else "mean ± 95% CI",
                "value": value,
                "error": error,
                "n_pockets": n,
            })

    ranks = load_ranks()
    for table in TABLES:
        log_table(table, results, ranks)

    logging.info(
        "\nPC SE: median ± MAD across pockets; all other metrics: mean ± 95% CI across pockets."
        "\nRank: average rank by closeness to the training-set property distribution."
        "\nSampling times are not stored in generated_data/ and are not exported."
    )

    with (OUTPUT_PATH / "main_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    logging.info(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
