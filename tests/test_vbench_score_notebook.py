"""Scoring notebook stays aligned with the builder, including the anatomy timm shim."""

import ast
import importlib.util
import json
import sys
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
        if "def shim_timm_pil_interp(" in text:
            return text
    raise AssertionError("shim_timm_pil_interp is missing from the scoring notebook")


def function_source(name):
    text = helpers_source()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(name)


def load_anatomy(namespace):
    tree = ast.parse(helpers_source())
    keep_def = {
        "shim_timm_pil_interp",
        "pip_would_change_torch",
        "install_pinned",
        "install_anatomy_stack",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "ANATOMY_PACKAGES" in names:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_def:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace.setdefault("print", lambda *args, **kwargs: None)
    namespace.setdefault("Path", Path)
    exec(compile(module, "<anatomy>", "exec"), namespace)
    return namespace


def install_fake_timm(transforms):
    saved = {name: sys.modules.get(name) for name in ("timm", "timm.data", "timm.data.transforms")}
    timm = types.ModuleType("timm")
    data = types.ModuleType("timm.data")
    timm.data = data
    data.transforms = transforms
    sys.modules["timm"] = timm
    sys.modules["timm.data"] = data
    sys.modules["timm.data.transforms"] = transforms
    return saved


def restore_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class ScoringNotebookTests(unittest.TestCase):
    def test_notebook_matches_builder(self):
        spec = importlib.util.spec_from_file_location("build_rq1_notebooks", BUILDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = NOTEBOOK.read_text(encoding="utf-8")
        module.build_scoring_notebook()
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), before)

    def test_notebook_code_parses(self):
        for cell in notebook_cells():
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

    def test_anatomy_shims_timm_before_mmdet_import_and_does_not_pin(self):
        text = code_text()
        self.assertNotIn("0.4.12", text)
        self.assertNotIn("anatomy_timm_constraint", text)
        self.assertNotIn("install_anatomy_timm", text)
        self.assertNotIn("def _pil_interp(", text)
        shim = function_source("shim_timm_pil_interp")
        self.assertIn("import timm.data.transforms as t", shim)
        self.assertIn('if not hasattr(t, "_pil_interp"):', shim)
        self.assertIn("t._pil_interp = t.str_to_pil_interp", shim)
        self.assertLess(
            shim.index("t._pil_interp = t.str_to_pil_interp"),
            shim.rindex('if not hasattr(t, "_pil_interp")'),
        )
        stack = function_source("install_anatomy_stack")
        self.assertLess(stack.index("shim_timm_pil_interp()"), stack.index("if not INSTALL_SCORING_DEPS:"))
        self.assertLess(stack.index("shim_timm_pil_interp()"), stack.index("install_pinned(ANATOMY_PACKAGES)"))
        self.assertNotIn('"-c"', function_source("install_pinned"))
        self.assertNotIn('"-c"', function_source("install_anatomy_mmcv"))
        self.assertNotIn('"-c"', function_source("install_anatomy_yolo"))
        anatomy = text.split('if spec["dimension"] == "Human_Anatomy":', 1)[1]
        self.assertLess(anatomy.index("install_anatomy_stack(checkout)"), anatomy.index("score_one("))

    def test_shim_aliases_the_renamed_function(self):
        transforms = types.ModuleType("timm.data.transforms")

        def str_to_pil_interp(mode):
            return "pil:" + mode

        transforms.str_to_pil_interp = str_to_pil_interp
        saved = install_fake_timm(transforms)
        try:
            load_anatomy({})["shim_timm_pil_interp"]()
        finally:
            restore_modules(saved)
        self.assertIs(transforms._pil_interp, str_to_pil_interp)
        self.assertEqual(transforms._pil_interp("bicubic"), "pil:bicubic")

    def test_shim_leaves_an_existing_pil_interp(self):
        transforms = types.ModuleType("timm.data.transforms")

        def old(mode):
            return "old:" + mode

        def renamed(mode):
            return "new:" + mode

        transforms._pil_interp = old
        transforms.str_to_pil_interp = renamed
        saved = install_fake_timm(transforms)
        try:
            load_anatomy({})["shim_timm_pil_interp"]()
        finally:
            restore_modules(saved)
        self.assertIs(transforms._pil_interp, old)

    def test_alias_exists_before_mmdet_install_and_when_installs_are_skipped(self):
        transforms = types.ModuleType("timm.data.transforms")

        def str_to_pil_interp(mode):
            return mode

        transforms.str_to_pil_interp = str_to_pil_interp
        saved = install_fake_timm(transforms)
        calls = []

        class Result:
            stdout = "Requirement already satisfied\n"
            stderr = ""
            returncode = 0

        class Subprocess:
            def run(self, command, **kwargs):
                self._record(command)
                return Result()

            def check_call(self, command, **kwargs):
                self._record(command)

            def _record(self, command):
                command = list(command)
                if any(str(part).startswith("mmdet") for part in command):
                    if not hasattr(transforms, "_pil_interp"):
                        raise AssertionError("mmdet install started before the timm alias")
                calls.append(command)

        try:
            namespace = load_anatomy({"sys": sys, "subprocess": Subprocess()})
            namespace["patch_anatomy_packages"] = lambda: None
            namespace["patch_anatomy_clip_config"] = lambda checkout: None
            namespace["install_anatomy_mmcv"] = lambda: None
            namespace["install_anatomy_yolo"] = lambda: None
            namespace["INSTALL_SCORING_DEPS"] = False
            namespace["install_anatomy_stack"](None)
            self.assertIs(transforms._pil_interp, str_to_pil_interp)
            self.assertFalse(any(str(part).startswith("mmdet") for command in calls for part in command))

            del transforms._pil_interp
            calls.clear()
            namespace["INSTALL_SCORING_DEPS"] = True
            namespace["install_anatomy_stack"](None)
        finally:
            restore_modules(saved)
        mmdet_commands = [
            command for command in calls if any(str(part).startswith("mmdet") for part in command)
        ]
        self.assertTrue(mmdet_commands)
        self.assertIs(transforms._pil_interp, str_to_pil_interp)

    def test_later_anatomy_installs_do_not_constrain_timm(self):
        calls = []

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

        namespace = load_anatomy({
            "sys": sys,
            "subprocess": Subprocess(),
        })
        namespace["install_pinned"](namespace["ANATOMY_PACKAGES"])
        installs = [command for command in calls if "--dry-run" not in command]
        self.assertEqual(len(installs), 1)
        self.assertIn("mmdet==3.0.0", installs[0])
        self.assertNotIn("-c", installs[0])
        self.assertFalse(any(part.startswith("timm") for part in installs[0]))


if __name__ == "__main__":
    unittest.main()
