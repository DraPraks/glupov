"""Scoring notebook stays aligned with the builder, including the anatomy timm pin."""

import ast
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "vbench_score_clips.ipynb"
BUILDER = ROOT / "scripts" / "build_rq1_notebooks.py"

VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def notebook_cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def code_text():
    return "\n".join("".join(cell["source"]) for cell in notebook_cells() if cell["cell_type"] == "code")


def helpers_source():
    for cell in notebook_cells():
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "def install_anatomy_timm(" in text:
            return text
    raise AssertionError("install_anatomy_timm is missing from the scoring notebook")


def function_source(name):
    text = helpers_source()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(name)


def load_anatomy(namespace):
    tree = ast.parse(helpers_source())
    keep_assign = {"ANATOMY_TIMM", "ANATOMY_PACKAGES"}
    keep_def = {
        "anatomy_timm_constraint",
        "installed_distribution_version",
        "timm_transforms_source_has_pil_interp",
        "install_anatomy_timm",
        "pip_would_change_torch",
        "install_pinned",
        "install_anatomy_stack",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if keep_assign.intersection(names):
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_def:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace.setdefault("print", lambda *args, **kwargs: None)
    exec(compile(module, "<anatomy>", "exec"), namespace)
    return namespace


class ScoringNotebookTests(unittest.TestCase):
    def test_notebook_matches_builder(self):
        spec = importlib.util.spec_from_file_location("build_rq1_notebooks", BUILDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = NOTEBOOK.read_text(encoding="utf-8")
        module.build_scoring_notebook()
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), before)

    def test_notebook_code_parses(self):
        for index, cell in enumerate(notebook_cells()):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell["source"]))

    def test_scored_dimensions_stay_put(self):
        text = code_text()
        for name in VBENCH_DIMENSIONS:
            self.assertIn(f'"{name}"', text)
        self.assertIn('VBENCH2_DIMENSIONS = ["Human_Anatomy"]', text)
        self.assertEqual(text.count('"not_applicable_multi_person"'), 2)
        self.assertIn("Human_Identity", text)
        self.assertIn("Human_Clothes", text)
        self.assertIn("already scored, left unchanged", text)

    def test_anatomy_pins_timm_before_mmdet_and_does_not_upgrade_it(self):
        text = helpers_source()
        stack = function_source("install_anatomy_stack")
        self.assertLess(stack.index("install_anatomy_timm()"), stack.index("install_pinned(ANATOMY_PACKAGES)"))
        self.assertIn('ANATOMY_TIMM = "0.4.12"', text)
        self.assertIn("Restart runtime once, then continue.", text)
        self.assertNotIn('pip install timm==0.9.16', text)
        for name in ("install_pinned", "install_anatomy_mmcv", "install_anatomy_yolo"):
            body = function_source(name)
            self.assertIn('"-c"', body)
            self.assertIn("anatomy_timm_constraint()", body)
        core = function_source("install_core_packages")
        self.assertIn('"timm"', core)
        self.assertNotIn("timm==", core)

    def test_newer_imported_timm_asks_for_one_restart_and_does_not_install(self):
        calls = []

        class Subprocess:
            def run(self, command, **kwargs):
                calls.append(list(command))
                raise AssertionError(command)

            def check_call(self, command, **kwargs):
                calls.append(list(command))
                raise AssertionError(command)

        fake_sys = types.SimpleNamespace(
            modules={
                "timm": types.SimpleNamespace(__version__="1.0.31"),
                "timm.data.transforms": types.SimpleNamespace(),
            },
            executable=sys.executable,
        )
        with tempfile.TemporaryDirectory() as tmp:
            namespace = {
                "CACHE_DIR": Path(tmp),
                "Path": Path,
                "sys": fake_sys,
                "subprocess": Subprocess(),
            }
            load_anatomy(namespace)
            with self.assertRaises(RuntimeError) as caught:
                namespace["install_anatomy_timm"]()
        self.assertIn("Restart runtime once, then continue.", str(caught.exception))
        self.assertEqual(calls, [])

    def test_unimported_timm_is_pinned_and_checked_before_mmdet(self):
        calls = []
        installed = {"timm": ""}

        class Result:
            stdout = "Requirement already satisfied\n"
            stderr = ""
            returncode = 0

        class Subprocess:
            def run(self, command, **kwargs):
                calls.append(list(command))
                return Result()

            def check_call(self, command, **kwargs):
                calls.append(list(command))
                installed["timm"] = "0.4.12"

        real_version = importlib.metadata.version
        real_find = importlib.util.find_spec

        def version(name):
            if name == "timm":
                if not installed["timm"]:
                    raise importlib.metadata.PackageNotFoundError(name)
                return installed["timm"]
            return real_version(name)

        def find_spec(name):
            if name == "timm.data.transforms":
                return types.SimpleNamespace(origin=str(transforms))
            return real_find(name)

        with tempfile.TemporaryDirectory() as tmp:
            transforms = Path(tmp) / "transforms.py"
            transforms.write_text("def _pil_interp(method):\n    return method\n", encoding="utf-8")
            namespace = {
                "CACHE_DIR": Path(tmp),
                "Path": Path,
                "sys": types.SimpleNamespace(modules={}, executable=sys.executable),
                "subprocess": Subprocess(),
            }
            load_anatomy(namespace)
            importlib.metadata.version = version
            importlib.util.find_spec = find_spec
            try:
                namespace["install_anatomy_timm"]()
                namespace["install_pinned"](namespace["ANATOMY_PACKAGES"])
            finally:
                importlib.metadata.version = real_version
                importlib.util.find_spec = real_find
            constraint = (Path(tmp) / "anatomy-timm-constraint.txt").read_text(encoding="utf-8")
        self.assertEqual(constraint, "timm==0.4.12\n")
        pin_commands = [command for command in calls if "timm==0.4.12" in command and "--dry-run" not in command]
        self.assertTrue(pin_commands)
        self.assertEqual(pin_commands[0][0:4], [sys.executable, "-m", "pip", "install"])
        mmdet = [command for command in calls if "mmdet==3.0.0" in command and "--dry-run" not in command]
        self.assertEqual(len(mmdet), 1)
        self.assertLess(calls.index(pin_commands[0]), calls.index(mmdet[0]))
        self.assertIn("-c", mmdet[0])
        self.assertTrue(any(part.endswith("anatomy-timm-constraint.txt") for part in mmdet[0]))
        for command in calls:
            if "--dry-run" in command:
                continue
            packages = [part for part in command if part.startswith("timm")]
            self.assertEqual(packages, ["timm==0.4.12"] if "timm==0.4.12" in command else [])

    def test_pin_that_lacks_the_symbol_stops_before_mmdet(self):
        class Result:
            stdout = ""
            stderr = ""
            returncode = 0

        class Subprocess:
            def run(self, command, **kwargs):
                return Result()

            def check_call(self, command, **kwargs):
                return None

        real_version = importlib.metadata.version
        real_find = importlib.util.find_spec

        def version(name):
            if name == "timm":
                return "0.4.12"
            return real_version(name)

        def find_spec(name):
            if name == "timm.data.transforms":
                return types.SimpleNamespace(origin=str(transforms))
            return real_find(name)

        with tempfile.TemporaryDirectory() as tmp:
            transforms = Path(tmp) / "transforms.py"
            transforms.write_text("def str_to_pil_interp(mode):\n    return mode\n", encoding="utf-8")
            namespace = {
                "CACHE_DIR": Path(tmp),
                "Path": Path,
                "sys": types.SimpleNamespace(modules={}, executable=sys.executable),
                "subprocess": Subprocess(),
            }
            load_anatomy(namespace)
            importlib.metadata.version = version
            importlib.util.find_spec = find_spec
            try:
                with self.assertRaises(RuntimeError) as caught:
                    namespace["install_anatomy_timm"]()
            finally:
                importlib.metadata.version = real_version
                importlib.util.find_spec = real_find
        self.assertIn("does not export timm.data.transforms._pil_interp", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
