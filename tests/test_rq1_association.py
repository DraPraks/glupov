"""Behavior of notebooks/rq1-aidrama.ipynb. Ratings live only in temp dirs."""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "rq1-aidrama.ipynb"
METRICS = ROOT / "data" / "metric_scores.csv"
TESTED = re.compile(
    r"^(?P<dimension>\S+)\s+n=(?P<n>\d+)\s+missing=(?P<miss>\d+)/(?P<elig>\d+)\s+"
    r"rho=(?P<rho>NA|-?\d+\.\d+)\s+CI=\[(?P<lo>NA|-?\d+\.\d+), (?P<hi>NA|-?\d+\.\d+)\]\s+"
    r"p_raw=(?P<praw>NA|\d+\.\d+)\s+p_holm=(?P<pholm>NA|\d+\.\d+)\s+"
    r"rho_median=(?P<med>NA|-?\d+\.\d+)$"
)
SKIPPED = re.compile(
    r"^(?P<dimension>\S+)\s+n=(?P<n>\d+)\s+missing=(?P<miss>\d+)/(?P<elig>\d+)\s+"
    r"status=(?P<status>\S+)$"
)


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


def spearman_stat(x, y):
    result = spearmanr(x, y)
    stat = getattr(result, "statistic", None)
    if stat is None:
        stat = result.correlation
    return float(stat)


def reference_association(x, y, seed, n_bootstrap, n_permutations):
    """Spearman, permutation, and bootstrap computed outside the notebook."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    scipy_rho = spearman_stat(x, y)
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    rho = float(rx @ ry / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    threshold = abs(scipy_rho) - 1e-12
    for _ in range(n_permutations):
        perm_rho = spearman_stat(x, rng.permutation(y))
        extreme += abs(perm_rho) >= threshold
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            boot.append(spearman_stat(x[idx], y[idx]))
    ci_low, ci_high = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975])
    return {
        "n": int(len(x)),
        "rho": rho,
        "scipy_rho": scipy_rho,
        "p_raw": (extreme + 1) / (n_permutations + 1),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def plant_stale(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rq1_spearman_results.csv"
    path.write_text("dimension,rho\nstale,1\n")
    return path


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
        self.assertIn("rq1_spearman_results.csv", text)

    def test_repo_ratings_absent_writes_nothing(self):
        self.assertFalse((ROOT / "data" / "human_ratings.csv").exists())
        before = set(os.listdir(ROOT))
        proc = run_notebook(None, None, cwd=ROOT / "notebooks", clear_data_env=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("human_ratings.csv", proc.stdout)
        self.assertIn("Enjoyment has not been collected.", proc.stdout)
        self.assertIn("does not invent ratings.", proc.stdout)
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)
        self.assertNotIn("Tested dimensions:", proc.stdout)
        self.assertNotIn("p_holm", proc.stdout)
        self.assertFalse((ROOT / "data" / "human_ratings.csv").exists())
        self.assertFalse((ROOT / "rq1_outputs").exists())
        self.assertEqual(set(os.listdir(ROOT)), before)
        self.assertEqual(proc.stdout.count("No correlation claims were written."), 1)

    def test_missing_metric_scores_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            proc = run_notebook(tmp / "data", tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("metric_scores.csv", proc.stdout)
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)

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
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)

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
        self.assertNotIn("rho=", proc.stdout)

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
            self.assertFalse((tmp / "out" / "rq1_spearman_results.csv").exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("unlabeled_pilot", proc.stdout)
        self.assertIn("not an independent set", proc.stdout)
        self.assertIn("human_sd", proc.stdout)
        self.assertIn("human_mean", proc.stdout)
        self.assertIn("Rating rows with missing enjoyment dropped: 1", proc.stdout)
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)
        self.assertNotIn("p_holm", proc.stdout)
        self.assertNotIn("Wrote ", proc.stdout)

    def test_repeated_real_source_id_skips_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            ids = ["drama_a", "drama_a", "drama_b", "drama_c", "drama_d", "drama_e"]
            frame = copy_metrics(data, source_ids=ids)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
            self.assertFalse((tmp / "out").exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("repeated source_id", proc.stdout)
        self.assertIn("drama_a", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)

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
        self.assertIn("Filled 6 missing source_id values from clip_manifest.csv.", proc.stdout)
        self.assertIn("unlabeled_pilot", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)

    def test_too_few_raters_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, source_ids=[f"drama_{i}" for i in range(6)])
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]), n_raters=2)
            proc = run_notebook(data, tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Eligible clips: 0", proc.stdout)
        self.assertIn("found 0", proc.stdout)
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)

    def test_non_integer_enjoyment_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, source_ids=[f"drama_{i}" for i in range(6)])
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                score_for=lambda i, rater, base: "3.5" if i == 0 and rater == 0 else base,
            )
            proc = run_notebook(data, tmp / "out")
            self.assertFalse((tmp / "out").exists())
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("integers from 1 to 5", proc.stderr)
        self.assertNotIn("rho=", proc.stdout)

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
            frame = copy_metrics(
                data,
                source_ids=["drama_0", "drama_1", None, "drama_3", "drama_4", "drama_5"],
            )
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            stale = plant_stale(out)
            proc = run_notebook(data, out)
            self.assertFalse(stale.exists())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Removed prior rq1_spearman_results.csv.", proc.stdout)
        self.assertIn("source_id is missing for at least one eligible clip", proc.stdout)
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)
        self.assertNotIn("p_holm", proc.stdout)

    def test_partial_source_id_is_filled_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(
                data,
                source_ids=["drama_0", "drama_1", "drama_2", None, None, None],
            )
            manifest = pd.DataFrame({
                "video_id": frame["video_id"],
                "source_id": [
                    "drama_0",
                    "drama_1",
                    "drama_2",
                    "unlabeled_pilot",
                    "unlabeled_pilot",
                    "unlabeled_pilot",
                ],
            })
            manifest.to_csv(data / "clip_manifest.csv", index=False)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Filled 3 missing source_id values from clip_manifest.csv.", proc.stdout)
        self.assertIn("drama_0", proc.stdout)
        self.assertIn("unlabeled_pilot", proc.stdout)
        self.assertIn("No correlation claims were written.", proc.stdout)
        self.assertNotIn("rho=", proc.stdout)

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
                    "source_id": ["drama_0", None, "drama_2", "drama_3", "drama_4", "drama_5"],
                }),
                "null source_id",
            ),
        }
        for name, (manifest, message) in cases.items():
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    data = tmp / "data"
                    frame = copy_metrics(data, source_ids=[None] * 6)
                    manifest.to_csv(data / "clip_manifest.csv", index=False)
                    write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
                    stale = plant_stale(tmp / "out")
                    proc = run_notebook(data, tmp / "out")
                    self.assertFalse(stale.exists())
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("rho=", proc.stdout)

    def test_existing_source_id_conflict_keeps_001_distinct_from_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            frame = copy_metrics(data, source_ids=["001", None, None, None, None, None])
            manifest = pd.DataFrame({
                "video_id": frame["video_id"],
                "source_id": ["1", "drama_1", "drama_2", "drama_3", "drama_4", "drama_5"],
            })
            manifest.to_csv(data / "clip_manifest.csv", index=False)
            write_ratings(data / "human_ratings.csv", list(frame["video_id"]))
            proc = run_notebook(data, tmp / "out")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("conflicts with metric_scores.csv", proc.stderr)
        self.assertIn(frame["video_id"].iloc[0], proc.stderr)
        self.assertNotIn("rho=", proc.stdout)

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
                    frame = copy_metrics(data, source_ids=[f"drama_{i}" for i in range(6)])
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
                self.assertNotIn("rho=", proc.stdout)

    def test_unique_drama_ids_report_spearman(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            out = tmp / "out"
            frame = copy_metrics(data, source_ids=["001", "1", "002", "2", "003", "3"])
            frame.loc[frame.index[0], "vbench_imaging_quality"] = np.nan
            frame.to_csv(data / "metric_scores.csv", index=False)
            ranks = frame["vbench_subject_consistency"].rank(method="average").astype(int).tolist()
            plans = {
                1: [1, 1, 1, 1, 1],
                2: [1, 2, 2, 2, 2],
                3: [2, 2, 3, 3, 3],
                4: [3, 3, 3, 4, 4],
                5: [4, 4, 4, 4, 5],
                6: [5, 5, 5, 5, 5],
            }
            write_ratings(
                data / "human_ratings.csv",
                list(frame["video_id"]),
                score_for=lambda i, rater, base: plans[ranks[i]][rater],
            )
            pd.DataFrame({
                "video_id": [frame["video_id"].iloc[0], frame["video_id"].iloc[0]],
                "source_id": ["001", "1"],
            }).to_csv(data / "clip_manifest.csv", index=False)
            proc = run_notebook(data, out)
            result_path = out / "rq1_spearman_results.csv"
            self.assertTrue(result_path.is_file(), proc.stdout + proc.stderr)
            saved = pd.read_csv(result_path)
            ratings = pd.read_csv(data / "human_ratings.csv")
            scored = pd.read_csv(data / "metric_scores.csv", dtype={"video_id": str, "source_id": str})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("001", proc.stdout)
        self.assertNotIn("repeated source_id", proc.stdout)
        self.assertNotIn("Filled ", proc.stdout)
        self.assertIn("Eligible clips N = 6", proc.stdout)
        self.assertIn("Holm family size = 6", proc.stdout)
        self.assertEqual(list(saved["dimension"]), EXPECTED_DIMENSIONS)
        means = ratings.groupby("video_id")["enjoyment"].mean()
        medians = ratings.groupby("video_id")["enjoyment"].median()
        y = means.loc[scored["video_id"]].to_numpy(dtype=float)
        y_median = medians.loc[scored["video_id"]].to_numpy(dtype=float)
        seed = notebook_int("SEED")
        n_bootstrap = notebook_int("N_BOOTSTRAP")
        n_permutations = notebook_int("N_PERMUTATIONS")
        p_family = []
        statuses = []
        for index, dimension in enumerate(EXPECTED_DIMENSIONS):
            row = saved.loc[saved["dimension"].eq(dimension)].iloc[0]
            statuses.append(row["status"])
            if row["status"] != "tested":
                self.assertTrue(pd.isna(row["p_raw"]))
                self.assertTrue(pd.isna(row["p_holm"]))
                p_family.append(1.0)
                continue
            reference = reference_association(
                scored[dimension], y, seed + index, n_bootstrap, n_permutations
            )
            self.assertAlmostEqual(reference["rho"], reference["scipy_rho"], places=9)
            self.assertAlmostEqual(row["rho"], reference["scipy_rho"], places=9)
            self.assertAlmostEqual(row["p_raw"], reference["p_raw"], places=9)
            self.assertAlmostEqual(row["ci_low"], reference["ci_low"], places=9)
            self.assertAlmostEqual(row["ci_high"], reference["ci_high"], places=9)
            self.assertEqual(int(row["n"]), reference["n"])
            x_values = np.asarray(scored[dimension], dtype=float)
            paired = np.isfinite(x_values) & np.isfinite(y_median)
            median_rho = spearman_stat(x_values[paired], y_median[paired])
            self.assertAlmostEqual(row["rho_median"], median_rho, places=9)
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
        self.assertEqual(dynamic["status"], "no_variation")
        self.assertEqual(int(dynamic["n_missing"]), 0)
        imaging = saved.loc[saved["dimension"].eq("vbench_imaging_quality")].iloc[0]
        self.assertEqual(imaging["status"], "tested")
        self.assertEqual(int(imaging["n_missing"]), 1)
        self.assertEqual(int(imaging["n"]), 5)
        subject = saved.loc[saved["dimension"].eq("vbench_subject_consistency")].iloc[0]
        self.assertAlmostEqual(subject["rho"], 1.0, places=9)
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
            if row["status"] == "tested":
                self.assertAlmostEqual(row["p_holm"], float(adjusted[index]), places=9)
            else:
                self.assertEqual(p_family[index], 1.0)

    def test_notebook_matches_builder(self):
        spec = importlib.util.spec_from_file_location(
            "build_rq1_notebooks", ROOT / "scripts" / "build_rq1_notebooks.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = NOTEBOOK.read_text()
        module.build_association_notebook()
        self.assertEqual(NOTEBOOK.read_text(), before)


if __name__ == "__main__":
    unittest.main()
