"""Build notebooks/vbench_score_clips.ipynb and notebooks/rq1-aidrama.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def src(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src(text),
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, "cells", len(cells))


def build_scoring_notebook() -> None:
    """Leave notebooks/vbench_score_clips.ipynb untouched.

    The committed file is the Colab notebook that scored Human Anatomy
    (lvis install and the timm _pil_interp alias). This builder is behind
    that notebook and must not replace it.
    """
    path = NB_DIR / "vbench_score_clips.ipynb"
    if path.is_file():
        print("left untouched", path)
        return
    cells: list[dict] = []
    cells.append(md("""
# VBench custom-input scoring for RQ1 clips

Run this on **Colab, A100 GPU, standard memory**. That runtime is about 40 GB of GPU memory and 84 GB of system RAM.

One raw clip is expanded to about 23 GB of frames in system RAM before VBench resizes it. That picks the machine:

- **A100, standard memory.** Enough RAM for the frames. Leave High-RAM off; that only spends more compute units.
- **L4** has about 53 GB of RAM, which is short of the frame buffer.
- **G4** is an RTX PRO 6000 Blackwell. Same generation as the RTX 5060 this already failed on. VBench's CUDA stack is not built for it.
- **H100** runs, and costs several times the A100 for six short clips.
- **Kaggle T4 x2** does not pool into one 32 GB GPU for these metrics, and a TPU cannot run this PyTorch code.

## Colab, nothing else to install

1. Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook → this file.
2. Runtime → Change runtime type → **A100 GPU**. Leave High-RAM unchecked.
3. Runtime → Run all.
4. When the last cell finishes, it downloads `rq1_vbench_csvs.zip`. Put those CSVs in the repo's `data/` folder. Add `human_ratings.csv` from Google Docs later, then run `rq1-aidrama.ipynb`.

This notebook scores the six VBench 1.0 dimensions that apply to these clips: subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. It also scores VBench 2.0 `Human_Anatomy` into `vbench2_human_anatomy`. A column that already has a score for every clip is left unchanged. Push `data/metric_scores.csv` before a second Colab run so those columns are not scored again.

`Human_Identity` and `Human_Clothes` are not scored. Both require one person, and these drama clips show several. Diversity and Multi-View Consistency need more than one clip per prompt. Those four are listed in `dimension_applicability.csv` and are absent from `metric_scores.csv`. `Human_Anatomy` looks at each person, so these multi-person clips are in scope. The confirmatory association notebook uses the dimensions this table marks scored, including `vbench2_human_anatomy`.

The notebook clones the public clip repo and the pinned VBench checkout, installs the Python packages, and downloads weights. Do not run VBench `setup.py`.

Missing scores stay missing. They are never written as zero. This notebook does not invent human enjoyment ratings.

Outputs:

- `clip_manifest.csv` — `video_id`, `path`, `source_id`, `duration`, `source_id_basis`
- `metric_scores.csv` — one row per clip, the six VBench 1.0 columns plus `vbench2_human_anatomy`
- `dimension_applicability.csv` — what ran, and why a dimension was skipped
"""))

    cells.append(code("""
from pathlib import Path
import json, math, os, platform, re, shutil, subprocess, sys
import numpy as np
import pandas as pd
from IPython.display import display

ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir()
TPU_RUNTIME = any(os.environ.get(name) for name in ("COLAB_TPU_ADDR", "TPU_NAME", "XRT_TPU_CONFIG"))
WORK_ROOT = Path("/content") if ON_COLAB else Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
CLIP_REPO = "https://github.com/DraPraks/glupov.git"
VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
RUN_SCORING = os.environ.get("RQ1_RUN_SCORING", "1") != "0"
HOSTED = ON_COLAB or ON_KAGGLE
INSTALL_SCORING_DEPS = os.environ.get("RQ1_INSTALL_DEPS", "1" if HOSTED else "0") == "1"
SOURCE_ID = "unlabeled_pilot"
SOURCE_ID_BASIS = "provisional_unlabeled"

A100_SWITCH = (
    "Runtime → Change runtime type → A100 GPU. Leave High-RAM off. "
    "Do not pick L4, G4, H100, T4, or a TPU."
)


def system_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 ** 2)
    return float("nan")


def require_colab_a100() -> None:
    if not RUN_SCORING:
        print("RQ1_RUN_SCORING=0: GPU check skipped.")
        return
    if TPU_RUNTIME:
        raise RuntimeError(
            "This runtime is a TPU. VBench is PyTorch CUDA and will not run here. " + A100_SWITCH
        )
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. " + A100_SWITCH)
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    ram_gb = system_ram_gb()
    print(f"GPU: {name}")
    print(f"Compute capability: {major}.{minor}")
    print(f"GPU memory: {vram_gb:.1f} GB")
    print(f"System RAM: {ram_gb:.1f} GB")
    if major >= 12:
        raise RuntimeError(
            f"{name} is Blackwell (compute capability {major}.{minor}). "
            "VBench's CUDA stack is not built for that generation. " + A100_SWITCH
        )
    if ram_gb < 60 or vram_gb < 35:
        raise RuntimeError(
            f"This runtime has {vram_gb:.0f} GB of GPU memory and {ram_gb:.0f} GB of system RAM. "
            "A full-resolution clip needs about 23 GB of system RAM before VBench resizes it. "
            + A100_SWITCH
        )
    if vram_gb >= 70:
        print("Larger than a standard A100. Scoring will run. Standard A100 spends fewer compute units.")
    else:
        print("A100-class GPU confirmed.")


def directory_has_video(root: Path) -> bool:
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}:
            if {"vbench_raw", "staged_clips", "rq1_outputs", "rq1_outputs_demo"} & set(path.parts):
                continue
            return True
    return False


def ensure_clip_dir() -> Path:
    override = os.environ.get("RQ1_CLIPS_DIR")
    if override:
        path = Path(override).resolve()
        if not directory_has_video(path):
            raise FileNotFoundError(f"RQ1_CLIPS_DIR has no videos: {path}")
        return path
    candidates = [
        Path.cwd() / "dataset",
        Path.cwd().parent / "dataset",
        WORK_ROOT / "glupov" / "dataset",
        WORK_ROOT / "dataset",
    ]
    if ON_KAGGLE:
        candidates.append(Path("/kaggle/input"))
    for candidate in candidates:
        if directory_has_video(candidate):
            return candidate.resolve()
    dest = WORK_ROOT / "glupov"
    print("Cloning clip repo", CLIP_REPO)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    subprocess.run(
        ["git", "clone", "--depth", "1", CLIP_REPO, str(dest)],
        check=True, env=env,
    )
    dataset = dest / "dataset"
    if not directory_has_video(dataset):
        raise FileNotFoundError(f"Clone succeeded but {dataset} has no videos.")
    return dataset.resolve()


require_colab_a100()
CLIPS_DIR = ensure_clip_dir()
if os.environ.get("RQ1_DATA_DIR"):
    DATA_DIR = Path(os.environ["RQ1_DATA_DIR"]).resolve()
elif HOSTED:
    DATA_DIR = (WORK_ROOT / "rq1_data").resolve()
else:
    repo_guess = CLIPS_DIR.parent if CLIPS_DIR.name == "dataset" else Path.cwd()
    DATA_DIR = (repo_guess / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get("RQ1_VBENCH_CACHE"):
    CACHE_DIR = Path(os.environ["RQ1_VBENCH_CACHE"]).resolve()
elif HOSTED:
    CACHE_DIR = (WORK_ROOT / "rq1-vbench").resolve()
else:
    CACHE_DIR = (Path.home() / ".cache" / "rq1-vbench").resolve()
REPO_ROOT = CLIPS_DIR.parent if CLIPS_DIR.name == "dataset" else CLIPS_DIR.parent

VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
VBENCH2_DIMENSIONS = ["Human_Anatomy"]

def metric_column(benchmark: str, dimension: str) -> str:
    if benchmark == "vbench":
        return "vbench_" + dimension
    return "vbench2_" + dimension.lower()

DIMENSION_SPECS = [
    {"benchmark": "vbench", "dimension": name, "column": metric_column("vbench", name)}
    for name in VBENCH_DIMENSIONS
] + [
    {"benchmark": "vbench2", "dimension": name, "column": metric_column("vbench2", name)}
    for name in VBENCH2_DIMENSIONS
]
DIMENSIONS = [spec["column"] for spec in DIMENSION_SPECS]
EXCLUDED_DIMENSIONS = [
    {
        "benchmark": "vbench2",
        "dimension": "Human_Identity",
        "single_clip": True,
        "status": "not_applicable_multi_person",
        "reason": (
            "VBench-2.0 anchors identity to one face in the opening frame and "
            "drops a clip with fewer than 20 single-face frames. "
            "These drama clips show several people, so the dimension is not scored."
        ),
    },
    {
        "benchmark": "vbench2",
        "dimension": "Human_Clothes",
        "single_clip": True,
        "status": "not_applicable_multi_person",
        "reason": (
            "VBench-2.0 asks whether only one person is present throughout and "
            "does not score clothes otherwise. These drama clips show several people."
        ),
    },
    {
        "benchmark": "vbench2",
        "dimension": "Diversity",
        "single_clip": False,
        "status": "not_applicable_single_clip",
        "reason": (
            "VBench-2.0 custom input needs >=20 videos for one prompt "
            "(prompt-index.mp4, index 0..19). One raw clip has no Diversity score."
        ),
    },
    {
        "benchmark": "vbench2",
        "dimension": "Multi-View_Consistency",
        "single_clip": False,
        "status": "not_applicable_single_clip",
        "reason": (
            "VBench-2.0 custom input expects multiple views in their own folder. "
            "One raw clip is not a multi-view set."
        ),
    },
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}
MANIFEST_CSV = DATA_DIR / "clip_manifest.csv"
METRICS_CSV = DATA_DIR / "metric_scores.csv"
APPLICABILITY_CSV = DATA_DIR / "dimension_applicability.csv"
VBENCH_RAW_DIR = DATA_DIR / "vbench_raw"
STAGED_DIR = DATA_DIR / "staged_clips"

print("REPO_ROOT =", REPO_ROOT)
print("CLIPS_DIR =", CLIPS_DIR)
print("DATA_DIR =", DATA_DIR)
print("Dimensions:", ", ".join(DIMENSIONS))
"""))

    cells.append(md("""
## Helpers
"""))

    cells.append(code("""
def coerce_score(raw):
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        if raw.strip().lower() in {"nan", "none", "null"}:
            return None
        try:
            raw = float(raw)
        except ValueError:
            return None
    if isinstance(raw, dict):
        for key in ("score", "video_results", "final_score"):
            if key in raw:
                return coerce_score(raw[key])
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return coerce_score(raw[0])
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_dimension_payload(payload):
    records = []
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and ("video_path" in item or "video_results" in item):
                records.append(item)
            elif isinstance(item, list):
                records.extend(sub for sub in item if isinstance(sub, dict))
    found = {}
    for record in records:
        path = record.get("video_path") or record.get("video_name") or ""
        found[str(path)] = coerce_score(record.get("video_results", None))
    return found


def file_name_of(path_str: str) -> str:
    return str(path_str).replace("\\\\", "/").rstrip("/").split("/")[-1]


def lookup_score(parsed: dict, staged_name: str):
    staged_stem = Path(staged_name).stem
    for key, value in parsed.items():
        name = file_name_of(key)
        if name == staged_name or Path(name).stem == staged_stem:
            return value
    return None


def list_videos(root: Path):
    root = Path(root)
    if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
        return [root]
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if {"data", "vbench_raw", "staged_clips", "rq1_outputs"} & set(path.parts):
            continue
        found.append(path)
    return found


def probe_duration(path: Path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return float(proc.stdout.strip()), ""
            except ValueError:
                pass
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        if fps > 0 and frames > 0:
            return float(frames) / float(fps), ""
    except Exception:
        pass
    return float("nan"), "Duration unread (need ffprobe or cv2)."


def make_video_id(stem: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "clip"
    video_id = base
    n = 2
    while video_id in used:
        video_id = f"{base}_{n}"
        n += 1
    used.add(video_id)
    return video_id


def applicability_table(per_dimension: dict | None = None) -> pd.DataFrame:
    per_dimension = per_dimension or {}
    rows = []
    for spec in DIMENSION_SPECS:
        extra = per_dimension.get(spec["dimension"], {})
        rows.append({
            "dimension": spec["dimension"],
            "column_name": spec["column"],
            "benchmark": spec["benchmark"],
            "custom_input_single_clip": True,
            "status": extra.get("status", "not_run"),
            "reason": extra.get(
                "reason",
                "Official custom_input scores one raw video on this dimension.",
            ),
        })
    for spec in EXCLUDED_DIMENSIONS:
        rows.append({
            "dimension": spec["dimension"],
            "column_name": "",
            "benchmark": spec["benchmark"],
            "custom_input_single_clip": spec["single_clip"],
            "status": spec["status"],
            "reason": spec["reason"],
        })
    return pd.DataFrame(rows)


def pip_would_change_torch(packages):
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", *packages],
        capture_output=True, text=True,
    )
    text = ((proc.stdout or "") + "\\n" + (proc.stderr or "")).lower()
    return "uninstalling torch" in text or "would install torch" in text or "downloading torch" in text


def install_packages(packages):
    if pip_would_change_torch(packages):
        raise RuntimeError(
            "Refusing to install " + ", ".join(packages)
            + " because pip would change the current torch build."
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def ensure_cuda_torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available. " + A100_SWITCH)
    return f"CUDA torch {torch.__version__} on {torch.cuda.get_device_name(0)}"


def ensure_decord():
    try:
        import decord  # noqa: F401
        return "decord import ok"
    except Exception as first_exc:
        if not INSTALL_SCORING_DEPS:
            raise RuntimeError(
                "decord is required for VBench video loading "
                f"({first_exc}). Set RQ1_INSTALL_DEPS=1 and rerun."
            ) from first_exc
        last_exc = first_exc
        for package in ("decord", "eva-decord"):
            try:
                install_packages([package])
                import decord  # noqa: F401
                return package + " installed"
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Could not install decord or eva-decord ({last_exc})") from last_exc


def shim_torchvision_write_video():
    import torchvision.io as video_io
    if not hasattr(video_io, "write_video"):
        def write_video(*args, **kwargs):
            raise RuntimeError(
                "torchvision.io.write_video is not available in this torchvision build"
            )
        video_io.write_video = write_video
        note = (
            "Added a torchvision.io.write_video shim. "
            "VBench-2.0 imports that name and does not call it."
        )
    else:
        note = "torchvision.io.write_video present"
    for name in list(sys.modules):
        if name == "vbench2" or name.startswith("vbench2."):
            del sys.modules[name]
    return note


def load_saved_metrics():
    candidates = [METRICS_CSV, REPO_ROOT / "data" / "metric_scores.csv", WORK_ROOT / "metric_scores.csv", Path.cwd() / "metric_scores.csv"]
    found = []
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        frame = pd.read_csv(path, dtype={"video_id": str, "source_id": str})
        frame["video_id"] = frame["video_id"].astype(str)
        filled = 0
        for column in DIMENSIONS:
            if column in frame.columns:
                filled += int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
        found.append((filled, path, frame))
        print("Found saved scores", path, "filled cells", filled)
    if not found:
        return None
    found.sort(key=lambda item: item[0], reverse=True)
    return found[0][2]


def install_core_packages():
    if not INSTALL_SCORING_DEPS:
        print("RQ1_INSTALL_DEPS=0: using packages already installed.")
        return
    required = [
        "opencv-python", "pyyaml", "easydict", "omegaconf", "tqdm",
        "einops", "timm", "scikit-image", "Pillow", "fvcore", "iopath",
        "ftfy", "regex", "openai-clip", "pyiqa", "gdown",
        "imageio", "imageio-ffmpeg", "huggingface_hub", "transformers==4.51.0",
        "tokenizers>=0.21,<0.22",
        "accelerate", "sentencepiece", "safetensors", "fairscale", "scenedetect",
    ]
    skipped = []
    for package in required:
        if pip_would_change_torch([package]):
            skipped.append(package)
            print("Skipping", package, "because pip would replace the CUDA torch build.")
            continue
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    if skipped:
        print("Packages left unchanged:", ", ".join(skipped))


ANATOMY_YOLO_COMMIT = "4826695a84ce370dad7fe4cb555a6c7162f3751d"
ANATOMY_YOLO_URL = "git+https://github.com/AILab-CVC/YOLO-World.git@" + ANATOMY_YOLO_COMMIT
ANATOMY_PACKAGES = [
    "setuptools>=77",
    "mmengine==0.10.5",
    "mmdet==3.0.0",
    "mmyolo==0.6.0",
    "supervision==0.19.0",
]


def anatomy_mmcv_requirement():
    import torch
    cuda = torch.version.cuda
    if not cuda or not torch.cuda.is_available():
        raise RuntimeError("Human Anatomy needs the CUDA torch that is already installed. " + A100_SWITCH)
    major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    version = f"2.2.0+cu.{cuda}.torch.{major_minor}"
    index = "https://wheels.astral.sh/simple/cu" + cuda.replace(".", "") + "/"
    return version, index


def install_pinned(packages):
    pending = []
    for package in packages:
        if pip_would_change_torch([package]):
            print("Skipping", package, "because pip would replace the CUDA torch build.")
            continue
        pending.append(package)
    if pending:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pending])


def install_anatomy_mmcv():
    import importlib.metadata
    version, index = anatomy_mmcv_requirement()
    try:
        installed = importlib.metadata.version("mmcv")
    except importlib.metadata.PackageNotFoundError:
        installed = ""
    if installed == version:
        print("mmcv", version, "already installed")
        return
    spec = "mmcv==" + version
    command = [sys.executable, "-m", "pip", "install", spec, "--extra-index-url", index]
    probe = subprocess.run(command + ["--dry-run"], capture_output=True, text=True)
    probe_text = ((probe.stdout or "") + "\\n" + (probe.stderr or "")).lower()
    if "uninstalling torch" in probe_text or "would install torch" in probe_text:
        command.append("--no-deps")
        print("Installing", spec, "without dependencies so the CUDA torch build stays.")
    else:
        print("Installing", spec)
    subprocess.check_call(command)


def installed_yolo_commit():
    import importlib.metadata
    try:
        dist = importlib.metadata.distribution("yolo-world")
    except importlib.metadata.PackageNotFoundError:
        return ""
    raw = dist.read_text("direct_url.json") or ""
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    vcs = payload.get("vcs_info") or {}
    return str(vcs.get("commit_id") or "")


def install_anatomy_yolo():
    found = installed_yolo_commit()
    if found == ANATOMY_YOLO_COMMIT:
        print("yolo-world", ANATOMY_YOLO_COMMIT, "already installed")
        return
    print("Installing YOLO-World", ANATOMY_YOLO_COMMIT, "without replacing mmyolo.")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
        "--no-build-isolation", ANATOMY_YOLO_URL,
    ])
    found = installed_yolo_commit()
    if found != ANATOMY_YOLO_COMMIT:
        raise RuntimeError(
            "YOLO-World install did not record commit " + ANATOMY_YOLO_COMMIT
            + ". Found " + (found or "no direct URL") + "."
        )
    if "yolo_world" in sys.modules:
        raise RuntimeError(
            "YOLO-World was reinstalled after it had been imported. "
            "Runtime → Restart runtime, then Run all."
        )


def patch_anatomy_packages():
    import site
    cap_old = "mmcv_maximum_version = '2.1.0'"
    cap_new = "mmcv_maximum_version = '2.3.0'"
    adafactor_old = "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)"
    adafactor_new = "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor, force=True)"
    notes = []
    loaded = [
        name for name in ("mmdet", "mmyolo", "mmengine.optim.optimizer.builder", "yolo_world")
        if name in sys.modules
    ]
    for root in site.getsitepackages():
        for rel in ("mmdet/__init__.py", "mmyolo/__init__.py"):
            path = Path(root, rel)
            if path.is_file() and cap_old in path.read_text(encoding="utf-8"):
                path.write_text(path.read_text(encoding="utf-8").replace(cap_old, cap_new), encoding="utf-8")
                notes.append(str(path))
        builder = Path(root, "mmengine/optim/optimizer/builder.py")
        if builder.is_file() and adafactor_old in builder.read_text(encoding="utf-8"):
            text = builder.read_text(encoding="utf-8")
            builder.write_text(text.replace(adafactor_old, adafactor_new, 1), encoding="utf-8")
            notes.append(str(builder))
    if notes and loaded:
        raise RuntimeError(
            "Anatomy patches were written after "
            + ", ".join(loaded)
            + " had already been imported. Runtime → Restart runtime, then Run all."
        )
    if notes:
        print("Patched anatomy packages:")
        print("\\n".join(notes))
    else:
        print("Anatomy package patches already present.")


def patch_anatomy_clip_config(checkout: Path):
    root = checkout / "VBench-2.0" / "vbench2" / "third_party" / "YOLO-World"
    private_clip = "/mnt/petrelfs/zhengdian/code/ckpt/clip-vit-base-patch32"
    public_clip = "openai/clip-vit-base-patch32"
    if not root.is_dir():
        raise RuntimeError(f"Missing YOLO-World configs under {root}")
    changed = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if private_clip in text:
            path.write_text(text.replace(private_clip, public_clip), encoding="utf-8")
            changed.append(str(path.relative_to(root)))
    remaining = []
    private_bytes = private_clip.encode("utf-8")
    for path in sorted(root.rglob("*")):
        if path.is_file() and private_bytes in path.read_bytes():
            remaining.append(str(path.relative_to(root)))
    if remaining:
        raise RuntimeError("Private CLIP path remains in " + ", ".join(remaining))
    if changed:
        print("Pointed CLIP at openai/clip-vit-base-patch32 in", ", ".join(changed))
    else:
        print("YOLO-World CLIP path already points at a public model.")


def install_anatomy_stack(checkout: Path):
    if not INSTALL_SCORING_DEPS:
        print("RQ1_INSTALL_DEPS=0: expecting the Human Anatomy packages to be installed.")
        patch_anatomy_packages()
        patch_anatomy_clip_config(checkout)
        return
    install_pinned(ANATOMY_PACKAGES)
    install_anatomy_mmcv()
    install_anatomy_yolo()
    patch_anatomy_packages()
    patch_anatomy_clip_config(checkout)


def ensure_vbench_checkout():
    dest = CACHE_DIR / "VBench"
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if not (dest / ".git").is_dir():
        if dest.exists():
            raise RuntimeError(f"{dest} exists but is not a git checkout.")
        subprocess.run(
            ["git", "clone", "https://github.com/Vchitect/VBench.git", str(dest)],
            check=True, env=env,
        )
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True, text=True, env=env,
    )
    current = (head.stdout or "").strip()
    if current != VBENCH_COMMIT:
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", VBENCH_COMMIT],
            check=True, env=env,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", VBENCH_COMMIT],
            check=True, env=env,
        )
        current = VBENCH_COMMIT
    return dest, current


def stage_for_vbench(src: Path, dest_mp4: Path) -> Path:
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    if suffix == ".gif":
        gif_dest = dest_mp4.with_suffix(".gif")
        if src.resolve() != gif_dest.resolve():
            shutil.copy2(src, gif_dest)
        return gif_dest
    if suffix == ".mp4":
        if src.resolve() != dest_mp4.resolve():
            shutil.copy2(src, dest_mp4)
        return dest_mp4
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"{src.name} is {suffix}. custom_input reads .mp4/.gif only, "
            "and ffmpeg is not on PATH."
        )
    copy = subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-c", "copy", str(dest_mp4)],
        capture_output=True, text=True,
    )
    if copy.returncode != 0:
        encode = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-an", str(dest_mp4)],
            capture_output=True, text=True,
        )
        if encode.returncode != 0:
            raise RuntimeError((encode.stderr or copy.stderr or "ffmpeg failed")[-800:])
    return dest_mp4


# Parser self-check: a returned 0 stays 0; missing stays missing.
_sample = [
    {"video_path": "a.mp4", "video_results": 0.0},
    {"video_path": "b.mp4", "video_results": None},
]
_parsed = parse_dimension_payload(_sample)
assert _parsed["a.mp4"] == 0.0
assert _parsed["b.mp4"] is None
print("Helpers ready.")
"""))

    cells.append(md("""
## 1. Build clip manifest

Pilot `source_id` is the shared provisional value `unlabeled_pilot`. Replace with real drama IDs before confirmatory independent-clip inference.
"""))

    cells.append(code("""
videos = list_videos(CLIPS_DIR)
if not videos:
    raise FileNotFoundError(
        f"No videos under {CLIPS_DIR}. Set RQ1_CLIPS_DIR or put mp4 files in dataset/."
    )

previous = {}
if MANIFEST_CSV.exists():
    old = pd.read_csv(MANIFEST_CSV, dtype=str)
    for record in old.to_dict("records"):
        if record.get("path"):
            previous[str(Path(record["path"]).resolve())] = record

used_ids: set[str] = set()
rows = []
duration_notes = []
for path in videos:
    path_str = str(path.resolve())
    prior = previous.get(path_str, {})
    if prior.get("video_id"):
        video_id = str(prior["video_id"])
        used_ids.add(video_id)
    else:
        video_id = make_video_id(path.stem, used_ids)
    # Prefer an already-edited source_id; otherwise keep the provisional shared id.
    if prior.get("source_id") and prior.get("source_id_basis") != SOURCE_ID_BASIS:
        source_id = str(prior["source_id"])
        basis = str(prior.get("source_id_basis") or "edited_manifest")
    elif prior.get("source_id") and str(prior.get("source_id")) != SOURCE_ID:
        source_id = str(prior["source_id"])
        basis = "edited_manifest"
    else:
        source_id = SOURCE_ID
        basis = SOURCE_ID_BASIS
    duration, note = probe_duration(path)
    if note:
        duration_notes.append(f"{video_id}: {note}")
    try:
        rel_path = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel_path = path_str
    rows.append({
        "video_id": video_id,
        "path": rel_path.replace("\\\\", "/"),
        "source_id": source_id,
        "duration": duration,
        "source_id_basis": basis,
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST_CSV, index=False)
print(f"Wrote {MANIFEST_CSV} ({len(manifest)} clips)")
if duration_notes:
    print("Duration notes:")
    print("\\n".join(duration_notes))
display(manifest)
"""))

    cells.append(md("""
## 2. Official custom-input scoring

Each VBench 1.0 dimension, and `Human_Anatomy`, is submitted alone with `mode='custom_input'`. Identity, clothes, Diversity, and Multi-View Consistency are not submitted. Set `RQ1_RUN_SCORING=0` to rebuild the metric table skeleton / reuse an existing file without calling VBench again.
"""))

    cells.append(code("""
def blank_metric_frame(manifest_df: pd.DataFrame) -> pd.DataFrame:
    frame = manifest_df[["video_id", "source_id"]].copy()
    frame["video_id"] = frame["video_id"].astype(str)
    frame["source_id"] = frame["source_id"].astype(str)
    for column in DIMENSIONS:
        frame[column] = np.nan
    return frame


def write_metric_frame(frame: pd.DataFrame) -> None:
    ordered = ["video_id", "source_id", *DIMENSIONS]
    out = frame[ordered].copy()
    for column in DIMENSIONS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.to_csv(METRICS_CSV, index=False)


def score_one(benchmark, dimension, column, staged_by_id, full_json):
    import torch
    run_name = column
    VBENCH_RAW_DIR.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        if benchmark == "vbench2":
            os.chdir(VBENCH2_HOME)
        if benchmark == "vbench":
            from vbench import VBench
            bench = VBench("cuda", str(full_json), str(VBENCH_RAW_DIR))
        else:
            from vbench2 import VBench2
            bench = VBench2("cuda", str(full_json), str(VBENCH_RAW_DIR))
        kwargs = dict(
            videos_path=str(STAGED_DIR),
            name=run_name,
            dimension_list=[dimension],
            mode="custom_input",
            local=False,
            read_frame=False,
            prompt_list=[],
        )
        if dimension == "imaging_quality":
            kwargs["imaging_quality_preprocessing_mode"] = "longer"
        bench.evaluate(**kwargs)
    finally:
        os.chdir(previous_cwd)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result_path = VBENCH_RAW_DIR / f"{run_name}_eval_results.json"
    if not result_path.is_file():
        raise RuntimeError(f"evaluate() did not write {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if dimension not in payload:
        raise RuntimeError(f"{result_path} has no '{dimension}' entry. Keys: {list(payload)}")
    parsed = parse_dimension_payload(payload[dimension])
    per_video = {}
    missing = []
    for video_id, staged in staged_by_id.items():
        value = lookup_score(parsed, staged.name) if parsed else None
        per_video[video_id] = value
        if value is None:
            missing.append(video_id)
    reason = f"Read per-video custom_input scores from {result_path.name}."
    if missing:
        reason += " No score returned for: " + ", ".join(missing) + "."
    if not parsed:
        reason = f"{result_path.name} had no per-video score records."
    return per_video, reason


metrics = blank_metric_frame(manifest)
saved_metrics = load_saved_metrics()
if saved_metrics is not None:
    saved_indexed = saved_metrics.drop_duplicates("video_id").set_index("video_id")
    for column in DIMENSIONS:
        if column in saved_indexed.columns:
            metrics[column] = metrics["video_id"].map(
                pd.to_numeric(saved_indexed[column], errors="coerce")
            )
per_dimension: dict = {}
blocker = ""

if not RUN_SCORING:
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV, dtype={"video_id": str, "source_id": str})
        for column in DIMENSIONS:
            if column in old.columns:
                metrics[column] = metrics["video_id"].map(old.set_index("video_id")[column])
        print("RQ1_RUN_SCORING=0: reused", METRICS_CSV)
    else:
        write_metric_frame(metrics)
        print("RQ1_RUN_SCORING=0: wrote empty metric skeleton", METRICS_CSV)
else:
    try:
        if not shutil.which("wget") or not shutil.which("unzip"):
            raise RuntimeError(
                "Official VBench weight download calls wget and unzip. "
                "Both exist on Colab. " + A100_SWITCH
            )
        torch_note = ensure_cuda_torch()
        checkout, commit = ensure_vbench_checkout()
        print(torch_note)
        print("VBench checkout", commit)
        entry = str(checkout)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        os.environ["VBENCH_CACHE_DIR"] = str(CACHE_DIR / "weights")
        os.environ["VBENCH2_CACHE_DIR"] = str(CACHE_DIR / "weights")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        VBENCH2_HOME = checkout / "VBench-2.0"
        vbench2_entry = str(VBENCH2_HOME)
        if vbench2_entry not in sys.path:
            sys.path.insert(0, vbench2_entry)
        print(shim_torchvision_write_video())
        install_core_packages()
        print(ensure_decord())
        full_vbench = checkout / "vbench" / "VBench_full_info.json"
        full_vbench2 = VBENCH2_HOME / "vbench2" / "VBench2_full_info.json"
        STAGED_DIR.mkdir(parents=True, exist_ok=True)
        staged_by_id = {}
        stage_errors = []
        for record in manifest.itertuples(index=False):
            src = (REPO_ROOT / record.path).resolve() if not Path(record.path).is_absolute() else Path(record.path)
            try:
                staged_by_id[str(record.video_id)] = stage_for_vbench(
                    src, STAGED_DIR / f"{record.video_id}.mp4"
                )
            except Exception as exc:
                stage_errors.append(f"{record.video_id}: {exc}")
        if stage_errors:
            print("Staging issues:\\n" + "\\n".join(stage_errors))
        if not staged_by_id:
            raise RuntimeError("No clips could be staged for VBench.")

        for spec in DIMENSION_SPECS:
            full_json = full_vbench if spec["benchmark"] == "vbench" else full_vbench2
            if not full_json.is_file():
                per_dimension[spec["dimension"]] = {
                    "status": "failed",
                    "reason": f"Missing full-info JSON: {full_json}",
                }
                continue
            already = pd.to_numeric(metrics[spec["column"]], errors="coerce")
            if already.notna().all():
                per_dimension[spec["dimension"]] = {
                    "status": "scored",
                    "reason": "Reused complete scores already stored in metric_scores.csv.",
                }
                print(" ", spec["dimension"], "already scored, left unchanged")
                continue
            try:
                if spec["dimension"] == "Human_Anatomy":
                    install_anatomy_stack(checkout)
                print("Scoring", spec["dimension"], "...")
                per_video, reason = score_one(
                    spec["benchmark"], spec["dimension"], spec["column"],
                    staged_by_id, full_json,
                )
                for video_id, value in per_video.items():
                    metrics.loc[metrics["video_id"] == video_id, spec["column"]] = (
                        np.nan if value is None else value
                    )
                per_dimension[spec["dimension"]] = {"status": "scored", "reason": reason}
                print(" ", spec["dimension"], "ok")
            except Exception as exc:
                msg = (type(exc).__name__ + ": " + str(exc))[-1000:]
                per_dimension[spec["dimension"]] = {"status": "failed", "reason": msg}
                print(" ", spec["dimension"], "FAILED:", msg[:300])
            write_metric_frame(metrics)
    except Exception as exc:
        blocker = str(exc)
        print("Scoring blocked:", blocker)
        write_metric_frame(metrics)

applicability_table(per_dimension).to_csv(APPLICABILITY_CSV, index=False)
write_metric_frame(metrics)
print("Wrote", METRICS_CSV)
print("Wrote", APPLICABILITY_CSV)
display(metrics)
display(pd.read_csv(APPLICABILITY_CSV))
if blocker:
    print("BLOCKER:", blocker)
"""))

    cells.append(md("""
## 3. Download the CSVs

The zip contains `clip_manifest.csv`, `metric_scores.csv`, and `dimension_applicability.csv`. On Colab the last lines start a browser download. Unzip into the repo's `data/` folder.

Human enjoyment is not in this zip. When the Google Docs ratings exist, add `data/human_ratings.csv` with columns `video_id`, `rater_id`, `enjoyment` (integers 1–5), replace `unlabeled_pilot` with real drama IDs, then run `notebooks/rq1-aidrama.ipynb`.
"""))

    cells.append(code("""
import zipfile
zip_path = DATA_DIR.parent / "rq1_vbench_csvs.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ("clip_manifest.csv", "metric_scores.csv", "dimension_applicability.csv"):
        path = DATA_DIR / name
        if path.is_file():
            archive.write(path, arcname=name)
            print("packed", name)
print("Wrote", zip_path)
if ON_COLAB:
    from google.colab import files
    files.download(str(zip_path))
else:
    print("CSVs are in", DATA_DIR)
"""))

    write_nb(NB_DIR / "vbench_score_clips.ipynb", cells)


def build_association_notebook() -> None:
    """Write the CSV-only RQ1 association notebook.

    Scoring stays in build_scoring_notebook(). This notebook does not invent ratings.
    """
    cells: list[dict] = []
    cells.append(md("""
# RQ1 association

Kendall's tau-b between the scored VBench columns and clip-level mean human enjoyment.

The confirmatory question is pairwise: when one clip has a higher dimension score than another, does it also tend to have higher enjoyment? Tau-b summarizes concordant pairs minus discordant pairs and adjusts for ties. The coefficient is `(P - Q) / sqrt((P + Q + T)(P + Q + U))`, from `scipy.stats.kendalltau(..., variant="b")`. P and Q are concordant and discordant clip pairs. T counts pairs tied only on the dimension score. U counts pairs tied only on mean enjoyment. Pairs tied on both are in neither count. With no ties this equals tau-a. Positive tau-b means higher scores tend to accompany higher enjoyment. The pairs are not extra independent clips. N is the number of clips.

Spearman is not estimated. Tau-b and Spearman are different scales and are not compared. A constant column makes tau-b undefined. That dimension stays in the Holm family as an internal p of 1, and its coefficient and p-values are reported as unavailable rather than as zero.

Reads `data/metric_scores.csv`, keeping `source_id` as text so values such as `001` and `1` stay distinct. When any `source_id` is blank, fills only those blanks from `data/clip_manifest.csv` after rejecting null, duplicate, or conflicting manifest IDs. Reads `data/human_ratings.csv` when that file exists (`video_id`, `rater_id`, `enjoyment` as integers 1–5).

The confirmatory family is every dimension `data/dimension_applicability.csv` marks `scored`: the six VBench 1.0 columns plus `vbench2_human_anatomy`. Identity, clothes, Diversity, and Multi-View Consistency stay out because that table marks them not applicable. `dynamic_degree` stays in the family; on this pilot it is constant, so tau-b is undefined.

This notebook does not score clips and does not invent ratings. If enjoyment ratings are missing, or confirmatory inference is blocked, it stops, writes no association claims, and removes a prior `rq1_kendall_results.csv` and any leftover `rq1_spearman_results.csv`. Non-numeric and infinite scores are rejected. Blank scores stay missing and are never filled with zero. Scores are not rounded or jittered before ties are judged. Clips with fewer than 3 ratings are excluded. Kendall's tau-b needs at least 4 eligible clips. Repeated `source_id` values, including provisional `unlabeled_pilot`, skip confirmatory inference until real drama IDs exist. `unlabeled_pilot` is not treated as a verified shared drama or as evidence that the clips are independent.

Bootstrap 95% percentile intervals resample paired clips with replacement. A replicate that makes either column constant is undefined: it is counted, omitted from the percentiles, and the notebook warns. The interval is reported only when at least 100 replicates are defined; otherwise the limits stay unavailable and the notebook warns. Intervals condition on the observed viewer panel and are not simultaneous family-wide intervals. The permutation test holds metric scores fixed and permutes mean enjoyment. Absolute tau-b is the two-sided statistic. Complete cases of size 7 or fewer (`7! = 5040`) use every enjoyment permutation. Larger samples use Monte Carlo. The Monte Carlo p-value is `(1 + count of |tau| at least the observed |tau|) / (1 + permutations)` and is never zero. Median enjoyment repeats tau-b as a sensitivity check, not as a second confirmatory test.

Defaults are seed 42, 2000 bootstrap replicates, and 4999 Monte Carlo permutations. Override them with `RQ1_SEED`, `RQ1_N_BOOTSTRAP`, and `RQ1_N_PERMUTATIONS`.
"""))
    cells.append(code("""
from pathlib import Path
import os
from itertools import permutations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests

DIMENSIONS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "vbench2_human_anatomy",
]
SCALE_MIN, SCALE_MAX = 1, 5
MIN_RATERS = 3
MIN_CLIPS = 4
ALPHA = 0.05
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_N_PERMUTATIONS = 4999
DEFAULT_SEED = 42
# Percentile intervals need this many defined replicates. Fewer stay unavailable.
MIN_VALID_BOOTSTRAPS = 100
# 7! = 5040, next to the default Monte Carlo budget. 8! = 40320, so larger n is Monte Carlo.
EXACT_MAX_N = 7
PROVISIONAL_SOURCE = "unlabeled_pilot"
RESULTS_NAME = "rq1_kendall_results.csv"
LEGACY_RESULTS_NAME = "rq1_spearman_results.csv"


def configured_int(name, default, minimum):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(name + " must be >= " + str(minimum) + ".")
    return value


N_BOOTSTRAP = configured_int("RQ1_N_BOOTSTRAP", DEFAULT_N_BOOTSTRAP, 1)
N_PERMUTATIONS = configured_int("RQ1_N_PERMUTATIONS", DEFAULT_N_PERMUTATIONS, 1)
SEED = configured_int("RQ1_SEED", DEFAULT_SEED, 0)


def stop(*lines):
    for line in lines:
        print(line)
    print("No association claims were written.")


def clear_prior_results():
    folder = output_dir()
    for name in (RESULTS_NAME, LEGACY_RESULTS_NAME):
        path = folder / name
        if path.is_file():
            path.unlink()
            print("Removed prior " + path.name + ".")


def repo_root():
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "data" / "metric_scores.csv").is_file():
            return candidate
    return cwd


def data_dir():
    override = os.environ.get("RQ1_DATA_DIR")
    if override:
        return Path(override).resolve()
    return repo_root() / "data"


def output_dir():
    override = os.environ.get("RQ1_OUT_DIR")
    if override:
        return Path(override).resolve()
    return repo_root() / "rq1_outputs"


def kendall_tau_b(x, y):
    result = kendalltau(np.asarray(x, float), np.asarray(y, float), variant="b")
    stat = getattr(result, "statistic", None)
    if stat is None:
        stat = result.correlation
    return float(stat)


def permutation_test(x, y, observed, rng):
    # Two-sided. Metric scores stay fixed; enjoyment is permuted.
    # The 1e-12 tolerance only absorbs floating-point noise around equality.
    threshold = abs(observed) - 1e-12
    if len(y) <= EXACT_MAX_N:
        extreme = 0
        total = 0
        for perm in permutations(np.asarray(y, float).tolist()):
            total += 1
            extreme += abs(kendall_tau_b(x, perm)) >= threshold
        return extreme / total, "exact"
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        extreme += abs(kendall_tau_b(x, rng.permutation(y))) >= threshold
    return (extreme + 1) / (N_PERMUTATIONS + 1), "monte_carlo"


def clean_id(series):
    text = series.astype("string").str.strip()
    return text.mask(series.isna() | text.isna() | text.eq(""))


def validated_manifest(manifest):
    if not {"video_id", "source_id"}.issubset(manifest.columns):
        raise ValueError("clip_manifest.csv needs video_id and source_id columns.")
    video = clean_id(manifest["video_id"])
    source = clean_id(manifest["source_id"])
    if video.isna().any():
        raise ValueError("clip_manifest.csv has a null video_id.")
    if video.duplicated().any():
        pairs = pd.DataFrame({
            "video_id": video.astype(str),
            "source_id": source.astype("string"),
        })
        nunique = pairs.groupby("video_id", dropna=False)["source_id"].nunique(dropna=False)
        if nunique.gt(1).any():
            raise ValueError(
                "clip_manifest.csv has conflicting source_id values for the same video_id."
            )
        raise ValueError("clip_manifest.csv has duplicate video_id values.")
    if source.isna().any():
        raise ValueError("clip_manifest.csv has a null source_id.")
    return pd.DataFrame({"video_id": video.astype(str), "source_id": source.astype(str)})


def fill_source_ids(metrics, folder):
    metrics = metrics.copy()
    if "source_id" not in metrics.columns:
        metrics["source_id"] = pd.Series(pd.NA, index=metrics.index, dtype="string")
    metrics["source_id"] = clean_id(metrics["source_id"])
    missing = metrics["source_id"].isna()
    if not missing.any():
        return metrics
    manifest_path = folder / "clip_manifest.csv"
    if not manifest_path.is_file():
        return metrics
    source = validated_manifest(pd.read_csv(manifest_path, dtype=str))
    known = metrics.loc[~missing, ["video_id", "source_id"]].merge(
        source, on="video_id", how="left", suffixes=("_metric", "_manifest")
    )
    if len(known):
        conflict = known["source_id_manifest"].notna() & known["source_id_metric"].ne(
            known["source_id_manifest"]
        )
        if conflict.any():
            ids = ", ".join(sorted(known.loc[conflict, "video_id"].astype(str)))
            raise ValueError(
                "clip_manifest.csv source_id conflicts with metric_scores.csv for: " + ids
            )
    filled = metrics.merge(source, on="video_id", how="left", suffixes=("", "_from_manifest"))
    take = filled["source_id"].isna() & filled["source_id_from_manifest"].notna()
    filled.loc[take, "source_id"] = filled.loc[take, "source_id_from_manifest"]
    print(
        "Filled "
        + str(int(take.sum()))
        + " missing source_id values from "
        + manifest_path.name
        + "."
    )
    return filled.drop(columns=["source_id_from_manifest"])


def load_metrics(folder):
    path = folder / "metric_scores.csv"
    if not path.is_file():
        stop(
            "Missing " + str(path) + ".",
            "This notebook only reads CSVs. It does not score clips.",
        )
        return None
    metrics = pd.read_csv(path, dtype={"video_id": str, "source_id": str})
    if "video_id" not in metrics.columns:
        raise ValueError(str(path) + " needs a video_id column.")
    video = clean_id(metrics["video_id"])
    if video.isna().any() or video.duplicated().any():
        raise ValueError("metric_scores.csv needs one non-null video_id per clip.")
    metrics = metrics.copy()
    metrics["video_id"] = video.astype(str)
    metrics = fill_source_ids(metrics, folder)
    print("Loaded " + path.name + ": " + str(len(metrics)) + " clips.")
    return metrics


def parse_score_column(series, name):
    text = series.astype("string").str.strip()
    blank = series.isna() | text.isna() | text.eq("")
    try:
        values = pd.to_numeric(text.mask(blank), errors="raise")
    except ValueError:
        raise ValueError(
            name + " has non-numeric scores. Blank cells stay missing."
        )
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(name + " contains infinite scores.")
    return values


def load_ratings(folder):
    path = folder / "human_ratings.csv"
    if not path.is_file():
        stop(
            "Missing " + str(path) + ".",
            "Expected columns: video_id, rater_id, enjoyment (integers 1-5).",
            "Enjoyment has not been collected. This notebook does not invent ratings.",
        )
        return None
    ratings = pd.read_csv(path, dtype={"video_id": str, "rater_id": str})
    if ratings.empty:
        stop(
            str(path) + " has no enjoyment ratings.",
            "Expected columns: video_id, rater_id, enjoyment (integers 1-5).",
            "Enjoyment has not been collected. This notebook does not invent ratings.",
        )
        return None
    missing_cols = {"video_id", "rater_id", "enjoyment"} - set(ratings.columns)
    if missing_cols:
        raise ValueError(
            "human_ratings.csv is missing columns: " + ", ".join(sorted(missing_cols))
        )
    if ratings["enjoyment"].dropna().empty:
        stop(
            str(path) + " has no enjoyment values.",
            "Expected integers 1-5. This notebook does not invent ratings.",
        )
        return None
    return ratings


def validate_ratings(ratings, metrics):
    if ratings[["video_id", "rater_id"]].isna().any().any():
        raise ValueError("video_id and rater_id must be filled on every rating row.")
    if ratings.duplicated(["video_id", "rater_id"]).any():
        raise ValueError("Duplicate video_id, rater_id rows. Resolve them before aggregating.")
    numeric = pd.to_numeric(ratings["enjoyment"], errors="coerce")
    invalid_text = ratings["enjoyment"].notna() & numeric.isna()
    present = numeric.dropna()
    non_integer = (present % 1) != 0
    out_of_range = ~present.between(SCALE_MIN, SCALE_MAX)
    if invalid_text.any() or non_integer.any() or out_of_range.any():
        raise ValueError(
            "enjoyment must be integers from 1 to 5, or blank if missing. "
            "This notebook does not impute ratings."
        )
    unknown = set(ratings["video_id"]) - set(metrics["video_id"])
    if unknown:
        raise ValueError(
            "Ratings for video_ids that are not in metric_scores.csv: "
            + ", ".join(sorted(unknown))
        )
    checked = ratings.copy()
    checked["enjoyment"] = numeric
    return checked


def clip_table(metrics, ratings):
    dropped = int(ratings["enjoyment"].isna().sum())
    valid = ratings.dropna(subset=["enjoyment"])
    summary = valid.groupby("video_id", as_index=False)["enjoyment"].agg(
        human_mean="mean",
        human_median="median",
        n_raters="count",
        human_sd="std",
    )
    clips = metrics.merge(summary, on="video_id", how="left", validate="one_to_one")
    clips["included"] = clips["n_raters"].fillna(0).ge(MIN_RATERS)
    print("Rating rows used: " + str(len(valid)))
    print("Distinct raters: " + str(valid["rater_id"].nunique()))
    print("Rating rows with missing enjoyment dropped: " + str(dropped))
    included = clips.loc[clips["included"]].copy()
    excluded = int((~clips["included"]).sum())
    print(
        "Eligible clips: "
        + str(len(included))
        + " (minimum "
        + str(MIN_RATERS)
        + " raters)"
    )
    print("Clips excluded for too few raters: " + str(excluded))
    if excluded:
        print(
            "Excluded video_id values: "
            + ", ".join(clips.loc[~clips["included"], "video_id"].astype(str))
        )
    if len(included):
        view_cols = ["video_id", "n_raters", "human_mean", "human_sd"]
        if "source_id" in included.columns:
            view_cols.insert(1, "source_id")
        view = included[view_cols].copy()
        view["n_raters"] = view["n_raters"].astype(int)
        view["human_mean"] = view["human_mean"].map(lambda v: format(v, ".3f"))
        view["human_sd"] = view["human_sd"].map(
            lambda v: "NA" if pd.isna(v) else format(v, ".3f")
        )
        print(view.to_string(index=False))
    return included


def confirmatory_block(included):
    if "source_id" not in included.columns:
        return (
            "Flag: no source_id column on the scored clips.",
            "Confirmatory inference stays off until real drama IDs exist.",
        )
    raw = included["source_id"]
    if raw.isna().any():
        return (
            "Flag: source_id is missing for at least one eligible clip.",
            "Confirmatory inference stays off until real drama IDs exist.",
        )
    sources = raw.astype(str).str.strip()
    if sources.eq("").any():
        return (
            "Flag: source_id is missing for at least one eligible clip.",
            "Confirmatory inference stays off until real drama IDs exist.",
        )
    n_pilot = int(sources.eq(PROVISIONAL_SOURCE).sum())
    if n_pilot:
        return (
            "Flag: source_id "
            + PROVISIONAL_SOURCE
            + " on "
            + str(n_pilot)
            + " of "
            + str(len(sources))
            + " eligible clips ("
            + str(int(sources.nunique()))
            + " distinct source_id value(s)).",
            "Drama grouping is unknown, so these clips are not an independent set.",
            "Confirmatory inference stays off until real drama IDs exist.",
        )
    if sources.duplicated().any():
        repeated = sorted(sources[sources.duplicated(keep=False)].unique())
        return (
            "Flag: repeated source_id values: " + ", ".join(repeated) + ".",
            "Confirmatory clip-level inference stays off until each eligible clip has its own drama ID, or the analysis is cluster-aware.",
        )
    return None


def association(x, y, seed):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n_unique_x = int(np.unique(x).size) if len(x) else 0
    n_unique_y = int(np.unique(y).size) if len(y) else 0
    out = {
        "n_complete_cases": int(len(x)),
        "n_unique_metric_values": n_unique_x,
        "n_unique_enjoyment_values": n_unique_y,
        "kendall_tau_b": np.nan,
        "ci_95_lower": np.nan,
        "ci_95_upper": np.nan,
        "p_permutation_raw": np.nan,
        "permutation_method": "not_applicable",
        "bootstrap_attempted_replicates": 0,
        "bootstrap_valid_replicates": 0,
        "bootstrap_undefined_replicates": 0,
        "analysis_status": "untestable",
    }
    if len(x) == 0:
        out["analysis_status"] = "all_missing"
        return out
    if len(x) < MIN_CLIPS:
        out["analysis_status"] = "too_few_clips"
        return out
    if n_unique_x < 2 or n_unique_y < 2:
        out["analysis_status"] = "no_variation"
        return out
    rng = np.random.default_rng(seed)
    tau_b = kendall_tau_b(x, y)
    p_raw, method = permutation_test(x, y, tau_b, rng)
    valid = []
    undefined = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            undefined += 1
            continue
        valid.append(kendall_tau_b(x[idx], y[idx]))
    if len(valid) >= MIN_VALID_BOOTSTRAPS:
        ci_low, ci_high = np.quantile(np.asarray(valid, float), [0.025, 0.975])
    else:
        ci_low, ci_high = np.nan, np.nan
    out.update(
        kendall_tau_b=tau_b,
        ci_95_lower=float(ci_low),
        ci_95_upper=float(ci_high),
        p_permutation_raw=float(p_raw),
        permutation_method=method,
        bootstrap_attempted_replicates=N_BOOTSTRAP,
        bootstrap_valid_replicates=len(valid),
        bootstrap_undefined_replicates=undefined,
        analysis_status="tested",
    )
    return out


def median_tau_b(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < MIN_CLIPS or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return kendall_tau_b(x, y)


def fmt(value, digits):
    if pd.isna(value):
        return "NA"
    return format(float(value), "." + str(digits) + "f")


def empty_result(dimension, n_eligible, status, n_missing):
    return {
        "dimension": dimension,
        "analysis_status": status,
        "n_eligible_clips": n_eligible,
        "n_complete_cases": 0,
        "n_missing_metric": n_missing,
        "n_unique_metric_values": 0,
        "n_unique_enjoyment_values": 0,
        "kendall_tau_b": np.nan,
        "ci_95_lower": np.nan,
        "ci_95_upper": np.nan,
        "p_permutation_raw": np.nan,
        "p_permutation_holm": np.nan,
        "permutation_method": "not_applicable",
        "bootstrap_attempted_replicates": 0,
        "bootstrap_valid_replicates": 0,
        "bootstrap_undefined_replicates": 0,
        "kendall_tau_b_median": np.nan,
        "reject_holm": False,
    }


def analyze(included):
    n_eligible = len(included)
    rows = []
    p_for_holm = []
    for i, dimension in enumerate(DIMENSIONS):
        if dimension not in included.columns:
            rows.append(empty_result(dimension, n_eligible, "missing_column", n_eligible))
            p_for_holm.append(1.0)
            continue
        x = parse_score_column(included[dimension], dimension)
        n_missing = int(x.isna().sum())
        result = association(x, included["human_mean"], SEED + i)
        result.update(
            dimension=dimension,
            n_missing_metric=n_missing,
            n_eligible_clips=n_eligible,
            kendall_tau_b_median=(
                median_tau_b(x, included["human_median"])
                if result["analysis_status"] == "tested"
                else np.nan
            ),
            p_permutation_holm=np.nan,
            reject_holm=False,
        )
        rows.append(result)
        # Untestable prespecified dimensions stay in the family as p = 1.
        # Their displayed p-values remain unavailable.
        p_for_holm.append(
            result["p_permutation_raw"] if result["analysis_status"] == "tested" else 1.0
        )
    _reject, adjusted, _a, _b = multipletests(p_for_holm, alpha=ALPHA, method="holm")
    for row, p_adj, reject in zip(rows, adjusted, _reject):
        if row["analysis_status"] == "tested":
            row["p_permutation_holm"] = float(p_adj)
            row["reject_holm"] = bool(reject)
    return pd.DataFrame(rows)


def print_results(results):
    n_eligible = int(results["n_eligible_clips"].iloc[0]) if len(results) else 0
    print("Eligible clips N = " + str(n_eligible) + ". n on each row is complete cases.")
    print(
        "Seed = "
        + str(SEED)
        + ". Bootstrap replicates = "
        + str(N_BOOTSTRAP)
        + ". Monte Carlo permutations = "
        + str(N_PERMUTATIONS)
        + ". Exact enumeration when complete cases <= "
        + str(EXACT_MAX_N)
        + "."
    )
    print("Tested dimensions:")
    tested = results.loc[results["analysis_status"].eq("tested")]
    if tested.empty:
        print("(none)")
    for row in tested.itertuples(index=False):
        if row.bootstrap_undefined_replicates:
            print(
                "Warning: "
                + row.dimension
                + " had "
                + str(int(row.bootstrap_undefined_replicates))
                + " undefined bootstrap replicates out of "
                + str(int(row.bootstrap_attempted_replicates))
                + "."
            )
        if row.bootstrap_valid_replicates < MIN_VALID_BOOTSTRAPS:
            print(
                "Warning: "
                + row.dimension
                + " bootstrap interval is unavailable ("
                + str(int(row.bootstrap_valid_replicates))
                + " valid replicates; at least "
                + str(MIN_VALID_BOOTSTRAPS)
                + " are required)."
            )
        print(
            row.dimension
            + "  n="
            + str(int(row.n_complete_cases))
            + "  missing="
            + str(int(row.n_missing_metric))
            + "/"
            + str(n_eligible)
            + "  kendall_tau_b="
            + fmt(row.kendall_tau_b, 3)
            + "  CI=["
            + fmt(row.ci_95_lower, 3)
            + ", "
            + fmt(row.ci_95_upper, 3)
            + "]  p_permutation_raw="
            + fmt(row.p_permutation_raw, 4)
            + "  p_permutation_holm="
            + fmt(row.p_permutation_holm, 4)
            + "  kendall_tau_b_median="
            + fmt(row.kendall_tau_b_median, 3)
            + "  permutation="
            + row.permutation_method
            + "  bootstrap_valid="
            + str(int(row.bootstrap_valid_replicates))
            + "  bootstrap_undefined="
            + str(int(row.bootstrap_undefined_replicates))
        )
    print("Skipped dimensions:")
    skipped = results.loc[~results["analysis_status"].eq("tested")]
    if skipped.empty:
        print("(none)")
    for row in skipped.itertuples(index=False):
        print(
            row.dimension
            + "  n="
            + str(int(row.n_complete_cases))
            + "  missing="
            + str(int(row.n_missing_metric))
            + "/"
            + str(n_eligible)
            + "  status="
            + row.analysis_status
        )
    print(
        "Holm family size = "
        + str(len(DIMENSIONS))
        + ". Untestable prespecified dimensions enter the correction as p = 1, "
        + "and their coefficient and p-values are reported as unavailable. "
        + "CIs are marginal paired-clip bootstrap 95% percentile intervals, "
        + "not simultaneous family-wide intervals. "
        + "They condition on the observed viewer panel. "
        + "p_permutation_raw is a two-sided exact or Monte Carlo permutation p "
        + "and is never zero. "
        + "kendall_tau_b_median repeats tau-b on median enjoyment and is not a second confirmatory test. "
        + "Associations are not causal."
    )


def write_results(results):
    folder = output_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RESULTS_NAME
    columns = [
        "dimension",
        "analysis_status",
        "n_eligible_clips",
        "n_complete_cases",
        "n_missing_metric",
        "n_unique_metric_values",
        "n_unique_enjoyment_values",
        "kendall_tau_b",
        "ci_95_lower",
        "ci_95_upper",
        "p_permutation_raw",
        "p_permutation_holm",
        "permutation_method",
        "bootstrap_attempted_replicates",
        "bootstrap_valid_replicates",
        "bootstrap_undefined_replicates",
        "kendall_tau_b_median",
        "reject_holm",
    ]
    results[columns].to_csv(path, index=False)
    print("Wrote " + str(path))


def main():
    clear_prior_results()
    folder = data_dir()
    print("DATA_DIR = " + str(folder))
    metrics = load_metrics(folder)
    if metrics is None:
        return
    ratings = load_ratings(folder)
    if ratings is None:
        return
    ratings = validate_ratings(ratings, metrics)
    included = clip_table(metrics, ratings)
    if len(included) < MIN_CLIPS:
        stop(
            "Kendall's tau-b needs at least "
            + str(MIN_CLIPS)
            + " eligible clips; found "
            + str(len(included))
            + ".",
            "Confirmatory inference was not run.",
        )
        return
    blocked = confirmatory_block(included)
    if blocked:
        stop(*blocked)
        return
    results = analyze(included)
    print_results(results)
    write_results(results)


main()
"""))
    write_nb(NB_DIR / "rq1-aidrama.ipynb", cells)


if __name__ == "__main__":
    build_association_notebook()
