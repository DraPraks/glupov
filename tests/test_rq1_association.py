"""Behavior of notebooks/rq1-aidrama.ipynb.

Ratings live only in temp dirs. They are code fixtures, not study data, and
these tests do not report a confirmatory RQ1 result.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import math
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "rq1-aidrama.ipynb"
METRICS = ROOT / "data" / "metric_scores.csv"


def notebook_code():
    nb = json.loads(NOTEBOOK.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if len(cells) != 1:
        raise AssertionError("expected one code cell")
    return "".join(cells[0]["source"])


def run_notebook(data, out, cwd=None, env_overrides=None, clear_data_env=False):
    env = os.environ.copy()
    if clear_data_env:
        env.pop("RQ1_DATA_DIR", None)
        env.pop("RQ1_OUT_DIR", None)
    else:
        env["RQ1_DATA_DIR"] = str(data)
        env["RQ1_OUT_DIR"] = str(out)
    for name in ("RQ1_N_BOOTSTRAP", "RQ1_N_PERMUTATIONS", "RQ1_SEED"):
        env.pop(name, None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", notebook_code()],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def write_ratings(path, video_ids, n_raters=5, score_for=None, extra_lines=None):
    lines = ["video_id,rater_id,enjoyment"]
    for i, video_id in enumerate(video_ids):
        base = (i % 5) + 1
        for rater in range(n_raters):
            score = base if score_for is None else score_for(i, rater, base)
            lines.append(f"{video_id},rater_{rater},{score}")
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n")


EXPECTED_DIMENSIONS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
]


def notebook_int(name):
    match = re.search(rf"^{name} = (\d+)$", notebook_code(), re.M)
    if not match:
        raise AssertionError("missing constant " + name)
    return int(match.group(1))


def kendall_tau_b(x, y):
    result = kendalltau(np.asarray(x, float), np.asarray(y, float), variant="b")
    stat = getattr(result, "statistic", None)
    if stat is None:
        stat = result.correlation
    return float(stat)


def pair_tau_b(x, y):
    """Tau-b from pair counts, independent of scipy.stats.kendalltau."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tied_x) * (concordant + discordant + tied_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def reference_association(x, y, seed, n_bootstrap, n_permutations, exact_max_n, min_valid):
    """Tau-b, permutation, and bootstrap computed outside the notebook."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    tau_b = kendall_tau_b(x, y)
    formula = pair_tau_b(x, y)
    rng = np.random.default_rng(seed)
    threshold = abs(tau_b) - 1e-12
    if len(y) <= exact_max_n:
        extreme = 0
        total = 0
        for perm in permutations(np.asarray(y, float).tolist()):
            total += 1
            extreme += abs(kendall_tau_b(x, perm)) >= threshold
        p_raw = extreme / total
        method = "exact"
    else:
        extreme = 0
        for _ in range(n_permutations):
            extreme += abs(kendall_tau_b(x, rng.permutation(y))) >= threshold
        p_raw = (extreme + 1) / (n_permutations + 1)
        method = "monte_carlo"
    valid = []
    undefined = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            undefined += 1
            continue
        valid.append(kendall_tau_b(x[idx], y[idx]))
    if len(valid) >= min_valid:
        ci_low, ci_high = np.quantile(np.asarray(valid, dtype=float), [0.025, 0.975])
    else:
        ci_low, ci_high = float("nan"), float("nan")
    return {
        "n": int(len(x)),
        "tau_b": tau_b,
        "formula": formula,
        "p_raw": float(p_raw),
        "method": method,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "valid": len(valid),
        "undefined": undefined,
    }


def plant_stale(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rq1_kendall_results.csv"
    path.write_text("dimension,kendall_tau_b\nstale,1\n")
    return path


def n_videos():
    return int(pd.read_csv(METRICS, dtype={"video_id": str}).shape[0])


def drama_ids(n=None):
    count = n_videos() if n is None else n
    return [f"drama_{i}" for i in range(count)]


def integer_ratings(total, n_raters=5):
    base, extra = divmod(int(total), n_raters)
    high = base + (1 if extra else 0)
    if base < 1 or high > 5:
        raise AssertionError("rating total " + str(total) + " is outside 1-5")
    return [base] * (n_raters - extra) + ([base + 1] * extra if extra else [])


def synthetic_metrics(dest, n, subject, **columns):
    frame = pd.DataFrame({
        "video_id": [f"clip_{i:02d}" for i in range(n)],
        "source_id": [f"drama_{i}" for i in range(n)],
    })
    for dimension in EXPECTED_DIMENSIONS:
        if dimension == "vbench_subject_consistency":
            frame[dimension] = subject
        else:
            frame[dimension] = columns.get(dimension, [1.0] * n)
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / "metric_scores.csv", index=False)
    return frame


def copy_metrics(dest, source_ids=None, drop_source=False):
    frame = pd.read_csv(METRICS, dtype={"video_id": str})
    if source_ids is not None:
        frame["source_id"] = source_ids
    if drop_source:
        frame = frame.drop(columns=["source_id"])
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / "metric_scores.csv", index=False)
    return frame


class Rq1AssociationNotebookTest(unittest.TestCase):
    def test_source_does_not_invent_ratings(self):
        text = notebook_code()
        lowered = text.lower()
        self.assertNotIn("demo_mode", lowered)
        self.assertNotIn("synthetic", lowered)
        self.assertNotIn("placeholder", lowered)
        self.assertNotIn(".uniform", text)
        self.assertNotIn(".normal", text)
        self.assertIn("default_rng", text)
        self.assertNotIn("human_ratings_template", text)
        self.assertNotIn("drop_duplicates", text)
        self.assertIn('dtype={"video_id": str, "source_id": str}', text)
        self.assertIn("contains infinite scores", text)
        self.assertEqual(text.count('errors="coerce"'), 1)
        self.assertIn('RESULTS_NAME = "rq1_kendall_results.csv"', text)
        self.assertIn('LEGACY_RESULTS_NAME = "rq1_spearman_results.csv"', text)
        self.assertNotIn("spearmanr", text)
        self.assertIn('variant="b"', text)
        self.assertIn("kendalltau", text)
        self.assertIn("Kendall's tau-b needs at least", text)

    def test_repo_ratings_absent_writes_nothing(self):
        self.assertFalse((ROOT / "data" / "human_ratings.csv").exists())
        before = set(os.listdir(ROOT))
        proc = run_notebook(None, None, cwd=ROOT / "notebooks", clear_data_env=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("human_ratings.csv", proc.stdout)
        self.assertIn("Enjoyment has not been collected.", proc.stdout)
        self.assertIn("does not invent ratings.", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)
        self.assertNotIn("Tested dimensions:", proc.stdout)
        self.assertNotIn("p_permutation_holm", proc.stdout)
        self.assertFalse((ROOT / "data" / "human_ratings.csv").exists())
        self.assertFalse((ROOT / "rq1_outputs").exists())
        self.assertEqual(set(os.listdir(ROOT)), before)
        self.assertEqual(proc.stdout.count("No association claims were written."), 1)

    def test_missing_metric_scores_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            proc = run_notebook(tmp / "data", tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("metric_scores.csv", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_empty_ratings_file_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            copy_metrics(data)
            (data / "human_ratings.csv").write_text("video_id,rater_id,enjoyment\n")
            proc = run_notebook(data, tmp / "out")
            self.assertFalse((tmp / "out").exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no enjoyment ratings", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_blank_enjoyment_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data)
            rows = ["video_id,rater_id,enjoyment"]
            rows.extend(f"{video_id},rater_0," for video_id in frame["video_id"])
            (data / "human_ratings.csv").write_text("\n".join(rows) + "\n")
            proc = run_notebook(data, tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no enjoyment values", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_unlabeled_pilot_skips_confirmatory_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data)
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                extra_lines=[f"{frame['video_id'].iloc[0]},rater_9,"],
            )
            proc = run_notebook(data, tmp / "out")
            self.assertFalse((tmp / "out" / "rq1_kendall_results.csv").exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("unlabeled_pilot", proc.stdout)
        self.assertIn("not an independent set", proc.stdout)
        self.assertIn("human_sd", proc.stdout)
        self.assertIn("human_mean", proc.stdout)
        self.assertIn("Rating rows with missing enjoyment dropped: 1", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)
        self.assertNotIn("p_permutation_holm", proc.stdout)
        self.assertNotIn("Wrote ", proc.stdout)

    def test_repeated_real_source_id_skips_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            ids = ["drama_a", "drama_a"] + [f"drama_{i}" for i in range(n_videos() - 2)]
            frame = copy_metrics(data, source_ids=ids)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
            self.assertFalse((tmp / "out").exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("repeated source_id", proc.stdout)
        self.assertIn("drama_a", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_source_id_from_manifest_still_blocks_pilot_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, drop_source=True)
            manifest = frame[["video_id"]].copy()
            manifest["source_id"] = "unlabeled_pilot"
            manifest.to_csv(data / "clip_manifest.csv", index=False)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(
            "Filled " + str(n_videos()) + " missing source_id values from clip_manifest.csv.",
            proc.stdout,
        )
        self.assertIn("unlabeled_pilot", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_too_few_raters_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, source_ids=drama_ids())
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]), n_raters=2)
            proc = run_notebook(data, tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Eligible clips: 0", proc.stdout)
        self.assertIn("found 0", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_non_integer_enjoyment_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, source_ids=drama_ids())
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                score_for=lambda i, rater, base: "3.5" if i == 0 and rater == 0 else base,
            )
            proc = run_notebook(data, tmp / "out")
            self.assertFalse((tmp / "out").exists())
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("integers from 1 to 5", proc.stderr)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_duplicate_rater_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data)
            video_id = frame["video_id"].iloc[0]
            (data / "human_ratings.csv").write_text(
                "video_id,rater_id,enjoyment\n"
                f"{video_id},rater_0,3\n"
                f"{video_id},rater_0,4\n"
            )
            proc = run_notebook(data, tmp / "out")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Duplicate video_id, rater_id", proc.stderr)

    def test_missing_eligible_source_ids_skip_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            out = tmp / "out"
            missing_ids = drama_ids()
            missing_ids[2] = None
            frame = copy_metrics(data, source_ids=missing_ids)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            stale = plant_stale(out)
            proc = run_notebook(data, out)
            self.assertFalse(stale.exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Removed prior rq1_kendall_results.csv.", proc.stdout)
        self.assertIn("source_id is missing for at least one eligible clip", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)
        self.assertNotIn("p_permutation_holm", proc.stdout)

    def test_blocked_run_removes_legacy_spearman_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            out = tmp / "out"
            copy_metrics(data)
            kendall = plant_stale(out)
            spearman = out / "rq1_spearman_results.csv"
            spearman.write_text("dimension,rho\nstale,1\n")
            proc = run_notebook(data, out)
            self.assertFalse(kendall.exists())
            self.assertFalse(spearman.exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Removed prior rq1_kendall_results.csv.", proc.stdout)
        self.assertIn("Removed prior rq1_spearman_results.csv.", proc.stdout)
        self.assertIn("Enjoyment has not been collected.", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_partial_source_id_is_filled_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            partial_ids = drama_ids()
            partial_ids[-3:] = [None, None, None]
            frame = copy_metrics(data, source_ids=partial_ids)
            manifest_ids = drama_ids()
            manifest_ids[-3:] = ["unlabeled_pilot", "unlabeled_pilot", "unlabeled_pilot"]
            manifest = pd.DataFrame({
                "video_id": frame["video_id"],
                "source_id": manifest_ids,
            })
            manifest.to_csv(data / "clip_manifest.csv", index=False)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Filled 3 missing source_id values from clip_manifest.csv.", proc.stdout)
        self.assertIn("drama_0", proc.stdout)
        self.assertIn("unlabeled_pilot", proc.stdout)
        self.assertIn("No association claims were written.", proc.stdout)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_manifest_rejects_duplicate_null_and_conflicting_ids(self):
        videos = pd.read_csv(METRICS, dtype={"video_id": str})["video_id"].tolist()
        cases = {
            "null video_id": (
                pd.DataFrame({
                    "video_id": [videos[0], None],
                    "source_id": ["drama_0", "drama_x"],
                }),
                "null video_id",
            ),
            "duplicate video_id": (
                pd.DataFrame({
                    "video_id": [videos[0], videos[0]],
                    "source_id": ["drama_0", "drama_0"],
                }),
                "duplicate video_id",
            ),
            "conflicting source_id": (
                pd.DataFrame({
                    "video_id": [videos[0], videos[0]],
                    "source_id": ["drama_0", "drama_9"],
                }),
                "conflicting source_id",
            ),
            "null source_id": (
                pd.DataFrame({
                    "video_id": videos,
                    "source_id": ["drama_0", None] + [f"drama_{i}" for i in range(2, len(videos))],
                }),
                "null source_id",
            ),
        }
        for name, (manifest, message) in cases.items():
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    data = tmp / "data"
                    frame = copy_metrics(data, source_ids=[None] * len(videos))
                    manifest.to_csv(data / "clip_manifest.csv", index=False)
                    write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
                    stale = plant_stale(tmp / "out")
                    proc = run_notebook(data, tmp / "out")
                    self.assertFalse(stale.exists())
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_existing_source_id_conflict_keeps_001_distinct_from_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, source_ids=["001"] + [None] * (n_videos() - 1))
            manifest = pd.DataFrame({
                "video_id": frame["video_id"],
                "source_id": ["1"] + [f"drama_{i}" for i in range(1, n_videos())],
            })
            manifest.to_csv(data / "clip_manifest.csv", index=False)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("conflicts with metric_scores.csv", proc.stderr)
        self.assertIn(frame["video_id"].iloc[0], proc.stderr)
        self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_non_numeric_and_infinite_scores_are_rejected(self):
        for label, value, message in (
            ("text", "bad", "non-numeric scores"),
            ("infinite", np.inf, "infinite scores"),
        ):
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    data = tmp / "data"
                    out = tmp / "out"
                    frame = copy_metrics(data, source_ids=drama_ids())
                    frame["vbench_subject_consistency"] = frame["vbench_subject_consistency"].astype(object)
                    frame.loc[frame.index[0], "vbench_subject_consistency"] = value
                    frame.to_csv(data / "metric_scores.csv", index=False)
                    write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
                    stale = plant_stale(out)
                    proc = run_notebook(data, out)
                    self.assertFalse(stale.exists())
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn("vbench_subject_consistency", proc.stderr)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("kendall_tau_b=", proc.stdout)

    def test_unique_drama_ids_report_kendall(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            out = tmp / "out"
            n = n_videos()
            frame = copy_metrics(data, source_ids=["001", "1"] + [f"drama_{i}" for i in range(2, n)])
            frame.loc[frame.index[0], "vbench_imaging_quality"] = np.nan
            frame.to_csv(data / "metric_scores.csv", index=False)
            subject = frame["vbench_subject_consistency"].to_numpy(dtype=float)
            self.assertEqual(len(np.unique(subject)), n)
            order = np.argsort(subject, kind="mergesort")
            rank_index = np.empty(n, dtype=int)
            rank_index[order] = np.arange(n)
            step = 20 // (n - 1)
            self.assertGreaterEqual(step, 1)
            totals = [5 + step * int(rank_index[i]) for i in range(n)]
            self.assertEqual(len(set(totals)), n)
            plans = [integer_ratings(total) for total in totals]
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                score_for=lambda i, rater, base: plans[i][rater],
            )
            pd.DataFrame({
                "video_id": [frame["video_id"].iloc[0], frame["video_id"].iloc[0]],
                "source_id": ["001", "1"],
            }).to_csv(data / "clip_manifest.csv", index=False)
            proc = run_notebook(data, out)
            result_path = out / "rq1_kendall_results.csv"
            self.assertTrue(result_path.is_file(), proc.stdout + proc.stderr)
            saved = pd.read_csv(result_path)
            ratings = pd.read_csv(data / "human_ratings.csv")
            scored = pd.read_csv(data / "metric_scores.csv", dtype={"video_id": str, "source_id": str})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("001", proc.stdout)
        self.assertNotIn("repeated source_id", proc.stdout)
        self.assertNotIn("Filled ", proc.stdout)
        self.assertIn("Eligible clips N = " + str(n), proc.stdout)
        self.assertIn("Holm family size = 6", proc.stdout)
        self.assertIn("vbench2_human_anatomy", scored.columns)
        for field in (
            "dimension",
            "analysis_status",
            "n_eligible_clips",
            "n_missing_metric",
            "n_unique_metric_values",
            "n_unique_enjoyment_values",
            "kendall_tau_b",
            "ci_95_lower",
            "ci_95_upper",
            "p_permutation_raw",
            "p_permutation_holm",
            "permutation_method",
            "bootstrap_valid_replicates",
            "bootstrap_undefined_replicates",
        ):
            self.assertIn(field, saved.columns)
        self.assertEqual(list(saved["dimension"]), EXPECTED_DIMENSIONS)
        means = ratings.groupby("video_id")["enjoyment"].mean()
        medians = ratings.groupby("video_id")["enjoyment"].median()
        y = means.loc[scored["video_id"]].to_numpy(dtype=float)
        y_median = medians.loc[scored["video_id"]].to_numpy(dtype=float)
        seed = notebook_int("DEFAULT_SEED")
        n_bootstrap = notebook_int("DEFAULT_N_BOOTSTRAP")
        n_permutations = notebook_int("DEFAULT_N_PERMUTATIONS")
        exact_max_n = notebook_int("EXACT_MAX_N")
        min_valid = notebook_int("MIN_VALID_BOOTSTRAPS")
        p_family = []
        statuses = []
        for index, dimension in enumerate(EXPECTED_DIMENSIONS):
            row = saved.loc[saved["dimension"].eq(dimension)].iloc[0]
            statuses.append(row["analysis_status"])
            self.assertEqual(int(row["n_eligible_clips"]), n)
            if row["analysis_status"] != "tested":
                self.assertTrue(pd.isna(row["kendall_tau_b"]))
                self.assertNotEqual(row["kendall_tau_b"], 0)
                self.assertTrue(pd.isna(row["p_permutation_raw"]))
                self.assertTrue(pd.isna(row["p_permutation_holm"]))
                self.assertEqual(row["permutation_method"], "not_applicable")
                p_family.append(1.0)
                continue
            reference = reference_association(
                scored[dimension],
                y,
                seed + index,
                n_bootstrap,
                n_permutations,
                exact_max_n,
                min_valid,
            )
            self.assertAlmostEqual(reference["tau_b"], reference["formula"], places=9)
            self.assertAlmostEqual(row["kendall_tau_b"], reference["tau_b"], places=9)
            self.assertAlmostEqual(row["p_permutation_raw"], reference["p_raw"], places=9)
            self.assertGreater(float(row["p_permutation_raw"]), 0.0)
            self.assertGreaterEqual(
                float(row["p_permutation_raw"]), 1 / (n_permutations + 1) - 1e-12
            )
            self.assertEqual(row["permutation_method"], "monte_carlo")
            self.assertEqual(row["permutation_method"], reference["method"])
            self.assertAlmostEqual(row["ci_95_lower"], reference["ci_low"], places=9)
            self.assertAlmostEqual(row["ci_95_upper"], reference["ci_high"], places=9)
            self.assertEqual(int(row["n_complete_cases"]), reference["n"])
            self.assertEqual(int(row["bootstrap_valid_replicates"]), reference["valid"])
            self.assertEqual(int(row["bootstrap_undefined_replicates"]), reference["undefined"])
            self.assertEqual(
                int(row["bootstrap_valid_replicates"]) + int(row["bootstrap_undefined_replicates"]),
                n_bootstrap,
            )
            x_values = np.asarray(scored[dimension], dtype=float)
            paired = np.isfinite(x_values) & np.isfinite(y_median)
            self.assertAlmostEqual(
                row["kendall_tau_b_median"],
                kendall_tau_b(x_values[paired], y_median[paired]),
                places=9,
            )
            p_family.append(reference["p_raw"])
        self.assertEqual(len(p_family), 6)
        self.assertEqual(
            [EXPECTED_DIMENSIONS[i] for i, status in enumerate(statuses) if status != "tested"],
            ["vbench_dynamic_degree"],
        )
        self.assertNotIn("vbench2_human_anatomy", set(saved["dimension"]))
        self.assertNotIn("vbench2_human_identity", set(saved["dimension"]))
        self.assertNotIn("vbench2_human_clothes", set(saved["dimension"]))
        dynamic = saved.loc[saved["dimension"].eq("vbench_dynamic_degree")].iloc[0]
        self.assertEqual(dynamic["analysis_status"], "no_variation")
        self.assertEqual(int(dynamic["n_missing_metric"]), 0)
        self.assertEqual(int(dynamic["n_unique_metric_values"]), 1)
        imaging = saved.loc[saved["dimension"].eq("vbench_imaging_quality")].iloc[0]
        self.assertEqual(imaging["analysis_status"], "tested")
        self.assertEqual(int(imaging["n_missing_metric"]), 1)
        self.assertEqual(int(imaging["n_complete_cases"]), n - 1)
        subject_row = saved.loc[saved["dimension"].eq("vbench_subject_consistency")].iloc[0]
        self.assertAlmostEqual(subject_row["kendall_tau_b"], 1.0, places=9)
        _reject, adjusted, _a, _b = multipletests(p_family, alpha=0.05, method="holm")
        tested_positions = [i for i, status in enumerate(statuses) if status == "tested"]
        _reject5, adjusted5, _a5, _b5 = multipletests(
            [p_family[i] for i in tested_positions], alpha=0.05, method="holm"
        )
        self.assertTrue(
            any(
                abs(float(adjusted[i]) - float(adjusted5[j])) > 1e-9
                for j, i in enumerate(tested_positions)
            )
        )
        for index, dimension in enumerate(EXPECTED_DIMENSIONS):
            row = saved.loc[saved["dimension"].eq(dimension)].iloc[0]
            if row["analysis_status"] == "tested":
                self.assertAlmostEqual(row["p_permutation_holm"], float(adjusted[index]), places=9)
            else:
                self.assertEqual(p_family[index], 1.0)
                self.assertTrue(pd.isna(row["p_permutation_holm"]))

    def test_strict_orders_are_plus_and_minus_one(self):
        for label, enjoyment in (("increasing", [1, 2, 3, 4]), ("reversed", [4, 3, 2, 1])):
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    data = tmp / "data"
                    values = [0.1, 0.2, 0.3, 0.4]
                    frame = synthetic_metrics(
                        data,
                        4,
                        values,
                        vbench_background_consistency=values,
                        vbench_motion_smoothness=values,
                        vbench_aesthetic_quality=values,
                        vbench_imaging_quality=values,
                    )
                    write_ratings(
                        data / "human_ratings.csv",
                        list(frame["video_id"]),
                        score_for=lambda i, rater, base, enjoyment=enjoyment: enjoyment[i],
                    )
                    proc = run_notebook(data, tmp / "out")
                    saved = pd.read_csv(tmp / "out" / "rq1_kendall_results.csv")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                expected = 1.0 if label == "increasing" else -1.0
                for dimension in EXPECTED_DIMENSIONS:
                    row = saved.loc[saved["dimension"].eq(dimension)].iloc[0]
                    if dimension == "vbench_dynamic_degree":
                        self.assertEqual(row["analysis_status"], "no_variation")
                        self.assertTrue(pd.isna(row["kendall_tau_b"]))
                        self.assertEqual(row["permutation_method"], "not_applicable")
                        continue
                    self.assertEqual(row["analysis_status"], "tested")
                    self.assertAlmostEqual(row["kendall_tau_b"], expected, places=9)
                    self.assertAlmostEqual(row["kendall_tau_b_median"], expected, places=9)
                    self.assertEqual(row["permutation_method"], "exact")
                    self.assertAlmostEqual(row["p_permutation_raw"], 2 / 24, places=9)
                    self.assertGreater(float(row["p_permutation_raw"]), 0.0)
                self.assertIn("undefined bootstrap", proc.stdout)

    def test_ties_match_scipy_and_constant_is_untestable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            subject = [0.1, 0.1, 0.3, 0.4, 0.4, 0.8]
            background = [1.0, np.nan, 1.0, 1.0, 1.0, 1.0]
            motion = [0.2, 0.4, 0.6, np.nan, np.nan, np.nan]
            frame = synthetic_metrics(
                data,
                6,
                subject,
                vbench_background_consistency=background,
                vbench_motion_smoothness=motion,
            )
            means = [1, 2, 2, 3, 3, 5]
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                score_for=lambda i, rater, base, means=means: means[i],
            )
            proc = run_notebook(data, tmp / "out")
            saved = pd.read_csv(tmp / "out" / "rq1_kendall_results.csv")
            ratings = pd.read_csv(data / "human_ratings.csv")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        y = ratings.groupby("video_id")["enjoyment"].mean().loc[frame["video_id"]].to_numpy(float)
        reference = reference_association(
            subject,
            y,
            notebook_int("DEFAULT_SEED"),
            notebook_int("DEFAULT_N_BOOTSTRAP"),
            notebook_int("DEFAULT_N_PERMUTATIONS"),
            notebook_int("EXACT_MAX_N"),
            notebook_int("MIN_VALID_BOOTSTRAPS"),
        )
        row = saved.loc[saved["dimension"].eq("vbench_subject_consistency")].iloc[0]
        self.assertEqual(row["analysis_status"], "tested")
        self.assertEqual(row["permutation_method"], "exact")
        self.assertAlmostEqual(row["kendall_tau_b"], kendall_tau_b(subject, y), places=9)
        self.assertAlmostEqual(row["kendall_tau_b"], pair_tau_b(subject, y), places=9)
        self.assertAlmostEqual(row["p_permutation_raw"], reference["p_raw"], places=9)
        self.assertGreater(float(row["p_permutation_raw"]), 0.0)
        self.assertEqual(int(row["bootstrap_undefined_replicates"]), reference["undefined"])
        self.assertGreater(int(row["bootstrap_undefined_replicates"]), 0)
        self.assertIn("undefined bootstrap", proc.stdout)
        background_row = saved.loc[saved["dimension"].eq("vbench_background_consistency")].iloc[0]
        self.assertEqual(background_row["analysis_status"], "no_variation")
        self.assertEqual(int(background_row["n_missing_metric"]), 1)
        self.assertTrue(pd.isna(background_row["kendall_tau_b"]))
        self.assertNotEqual(background_row["kendall_tau_b"], 0)
        motion_row = saved.loc[saved["dimension"].eq("vbench_motion_smoothness")].iloc[0]
        self.assertEqual(motion_row["analysis_status"], "too_few_clips")
        self.assertEqual(int(motion_row["n_missing_metric"]), 3)
        self.assertEqual(int(motion_row["n_complete_cases"]), 3)
        self.assertTrue(pd.isna(motion_row["kendall_tau_b"]))
        self.assertEqual(list(saved["dimension"]), EXPECTED_DIMENSIONS)

    def test_fixed_seed_reproduces_monte_carlo(self):
        frames = []
        env = {"RQ1_SEED": "7", "RQ1_N_BOOTSTRAP": "150", "RQ1_N_PERMUTATIONS": "100"}
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                data = tmp / "data"
                values = [0.1 * (i + 1) for i in range(8)]
                frame = synthetic_metrics(
                    data,
                    8,
                    values,
                    vbench_background_consistency=values,
                    vbench_motion_smoothness=values,
                    vbench_dynamic_degree=values,
                    vbench_aesthetic_quality=values,
                    vbench_imaging_quality=values,
                )
                write_ratings(
                    data / "human_ratings.csv",
                    list(frame["video_id"]),
                    score_for=lambda i, rater, base: min(5, 1 + (i % 5)),
                )
                proc = run_notebook(data, tmp / "out", env_overrides=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                saved = pd.read_csv(tmp / "out" / "rq1_kendall_results.csv")
            frames.append(saved)
        pd.testing.assert_frame_equal(frames[0], frames[1])
        self.assertTrue((frames[0]["permutation_method"] == "monte_carlo").all())
        self.assertTrue((frames[0]["p_permutation_raw"] > 0).all())
        self.assertTrue(frames[0]["ci_95_lower"].notna().all())

    def test_short_bootstrap_does_not_invent_an_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            values = [0.1 * (i + 1) for i in range(8)]
            frame = synthetic_metrics(
                data,
                8,
                values,
                vbench_background_consistency=values,
                vbench_motion_smoothness=values,
                vbench_dynamic_degree=values,
                vbench_aesthetic_quality=values,
                vbench_imaging_quality=values,
            )
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                score_for=lambda i, rater, base: min(5, 1 + (i % 5)),
            )
            proc = run_notebook(
                data,
                tmp / "out",
                env_overrides={"RQ1_N_BOOTSTRAP": "30", "RQ1_N_PERMUTATIONS": "40", "RQ1_SEED": "3"},
            )
            saved = pd.read_csv(tmp / "out" / "rq1_kendall_results.csv")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("bootstrap interval is unavailable", proc.stdout)
        self.assertTrue(saved["ci_95_lower"].isna().all())
        self.assertTrue(saved["ci_95_upper"].isna().all())
        self.assertTrue((saved["analysis_status"] == "tested").all())
        self.assertTrue((saved["p_permutation_raw"] > 0).all())
        self.assertTrue((saved["bootstrap_valid_replicates"] < notebook_int("MIN_VALID_BOOTSTRAPS")).all())

    def test_notebook_matches_builder(self):
        spec = importlib.util.spec_from_file_location(
            "build_rq1_notebooks", ROOT / "scripts" / "build_rq1_notebooks.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = NOTEBOOK.read_text()
        scoring = ROOT / "notebooks" / "vbench_score_clips.ipynb"
        scoring_before = scoring.read_bytes()
        module.build_association_notebook()
        self.assertEqual(NOTEBOOK.read_text(), before)
        module.build_scoring_notebook()
        self.assertEqual(scoring.read_bytes(), scoring_before)


if __name__ == "__main__":
    unittest.main()
