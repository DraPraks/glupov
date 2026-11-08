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
    cells: list[dict] = []
    cells.append(md("""
# VBench custom-input scoring for RQ1 clips

Run this on **Colab, A100 GPU, standard memory**. That runtime is about 40 GB of GPU memory and 84 GB of system RAM.

One raw clip is expanded to about 23 GB of frames in system RAM before VBench resizes it. `Human_Clothes` then loads LLaVA-Video-7B in bf16 (about 15 GB of weights plus a 64-frame video). Those two facts pick the machine:

- **A100, standard memory.** Enough RAM for the frames and enough GPU memory for LLaVA. Leave High-RAM off; that only spends more compute units.
- **L4** has about 53 GB of RAM, so the frames fit, but 22 GB of GPU memory is too tight for LLaVA-Video-7B.
- **G4** is an RTX PRO 6000 Blackwell. Same generation as the RTX 5060 this already failed on. VBench's CUDA stack is not built for it.
- **H100** runs, and costs several times the A100 for six short clips.
- **Kaggle T4 x2** does not pool into one 32 GB GPU for these metrics, and a TPU cannot run this PyTorch code.

## Colab, nothing else to install

1. Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook → this file.
2. Runtime → Change runtime type → **A100 GPU**. Leave High-RAM unchecked.
3. Runtime → Run all.
4. When the last cell finishes, it downloads `rq1_vbench_csvs.zip`. Put those CSVs in the repo's `data/` folder. Add `human_ratings.csv` from Google Docs later, then run `rq1-aidrama.ipynb`.

A column that already has a score for every clip is left unchanged. Push `data/metric_scores.csv` before a second Colab run so the six finished VBench columns are not scored again.

`Human_Clothes` loads LLaVA with PyTorch SDPA. This Colab torch build has no `flash_attn` package, and SDPA is the attention implementation that still answers in real words. Each question sent to LLaVA is printed, then the decoded answer. Before reading frames it sets Decord's bridge back to native, because `Human_Identity` switches that process-wide bridge to torch and `asnumpy()` then fails. A VBench result of `-1` is left blank: for clothes it means the clip is not one person throughout, and for identity it means fewer than 20 frames contained exactly one face. If every clothes clip is `-1`, the average is skipped instead of dividing by zero.

The notebook clones the public clip repo and the pinned VBench checkout, installs the Python packages, and downloads weights. Do not run VBench `setup.py`.

Missing scores stay missing. They are never written as zero. This notebook does not invent human enjoyment ratings.

Outputs:

- `clip_manifest.csv` — `video_id`, `path`, `source_id`, `duration`, `source_id_basis`
- `metric_scores.csv` — one row per clip, versioned VBench / VBench-2.0 columns
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
            "A full-resolution clip needs about 23 GB of system RAM, and Human_Clothes needs "
            "about 40 GB of GPU memory so LLaVA-Video-7B fits. " + A100_SWITCH
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
VBENCH2_DIMENSIONS = ["Human_Anatomy", "Human_Identity", "Human_Clothes"]

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
EXCLUDED_SINGLE_CLIP = [
    {
        "benchmark": "vbench2",
        "dimension": "Diversity",
        "reason": (
            "VBench-2.0 custom input needs >=20 videos for one prompt "
            "(prompt-index.mp4, index 0..19). One raw clip has no Diversity score."
        ),
    },
    {
        "benchmark": "vbench2",
        "dimension": "Multi-View_Consistency",
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
    for spec in EXCLUDED_SINGLE_CLIP:
        rows.append({
            "dimension": spec["dimension"],
            "column_name": "",
            "benchmark": spec["benchmark"],
            "custom_input_single_clip": False,
            "status": "not_applicable_single_clip",
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


def ensure_huggingface_cli():
    # Colab's huggingface-cli is a stub that exits before any download. VBench calls that name.
    bindir = Path.home() / ".local" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    script = bindir / "huggingface-cli"
    script.write_text(
        "#!/usr/bin/env python\\n"
        "import sys\\n"
        "from huggingface_hub import snapshot_download\\n"
        "args = sys.argv[1:]\\n"
        "if len(args) < 2 or args[0] != 'download':\\n"
        "    raise SystemExit('huggingface-cli shim only implements: download <repo> --local-dir <dir>')\\n"
        "repo = args[1]\\n"
        "local_dir = args[args.index('--local-dir') + 1]\\n"
        "repo_type = 'model'\\n"
        "if '--repo-type' in args:\\n"
        "    repo_type = args[args.index('--repo-type') + 1]\\n"
        "snapshot_download(repo_id=repo, repo_type=repo_type, local_dir=local_dir)\\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    return "huggingface-cli shim is first on PATH and downloads public repos without a token"


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
    try:
        if not pip_would_change_torch(["retinaface_pytorch==0.0.8"]):
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "retinaface_pytorch==0.0.8", "--no-deps",
            ])
        else:
            print("Skipping retinaface_pytorch because pip would replace torch.")
    except Exception as exc:
        print("retinaface_pytorch install failed. Human_Identity will record the import error.")
        print(exc)
    identity_deps = ["albumentations==1.4.18", "opencv-python-headless<4.11"]
    try:
        if pip_would_change_torch(identity_deps):
            print("Skipping albumentations because pip would replace torch.")
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *identity_deps])
    except Exception as exc:
        print("albumentations install failed. Human_Identity will record the import error.")
        print(exc)


def add_llava_import_root(checkout: Path) -> Path:
    llava_root = checkout / "VBench-2.0" / "vbench2" / "third_party" / "LLaVA_NeXT"
    if not llava_root.is_dir():
        raise FileNotFoundError(f"Bundled LLaVA-NeXT is missing: {llava_root}")
    entry = str(llava_root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return llava_root


def prefetch_retinaface() -> str:
    import torch
    url = "https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip"
    dest = Path(torch.hub.get_dir()) / "checkpoints" / "retinaface_resnet50_2020-07-20-f168fae3c.zip"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return f"retinaface weights already at {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    last = "not attempted"
    for attempt in range(1, 4):
        proc = subprocess.run(
            ["curl", "-L", "--retry", "5", "--retry-all-errors", "-o", str(partial), url],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and partial.is_file() and partial.stat().st_size > 1_000_000:
            partial.replace(dest)
            return f"retinaface weights downloaded to {dest}"
        last = (proc.stderr or proc.stdout or "curl failed")[-400:]
        print(f"retinaface download attempt {attempt} failed:", last)
    print("Human_Identity will try torch.hub next, which hit HTTP 504 on the last Colab run.")
    return last


def drop_imported_vbench() -> None:
    for name in list(sys.modules):
        if (
            name == "vbench2"
            or name.startswith("vbench2.")
            or name == "llava"
            or name.startswith("llava.")
        ):
            del sys.modules[name]


def use_pytorch_sdpa_for_llava(checkout: Path) -> str:
    # LLaVA-NeXT defaults to flash_attention_2. Colab's torch has no flash_attn wheel.
    # SDPA is built into this torch and is the implementation that still answers in words.
    builder = (
        checkout / "VBench-2.0" / "vbench2" / "third_party"
        / "LLaVA_NeXT" / "llava" / "model" / "builder.py"
    )
    text = builder.read_text(encoding="utf-8")
    old = 'attn_implementation="flash_attention_2"'
    new = 'attn_implementation="sdpa"'
    if old not in text:
        if new not in text:
            raise RuntimeError(f"LLaVA builder has no attention default to change: {builder}")
    else:
        builder.write_text(text.replace(old, new, 1), encoding="utf-8")
    drop_imported_vbench()
    return "LLaVA attention is sdpa. This torch build has no flash_attn package."


def fix_clothes_decord_bridge(checkout: Path) -> str:
    # Human Identity sets Decord's process-wide bridge to torch at import.
    # Clothes calls asnumpy(), which exists on a Decord array and not on a Tensor.
    path = checkout / "VBench-2.0" / "vbench2" / "human_clothes.py"
    text = path.read_text(encoding="utf-8")
    needle = "from decord import VideoReader, cpu\\n"
    if "import decord\\n" not in text:
        if needle not in text:
            raise RuntimeError("human_clothes.py has no decord import to extend")
        text = text.replace(needle, "import decord\\n" + needle, 1)
    start = text.find("def load_video(")
    end = text.find("\\ndef LLaVA_Video(")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("human_clothes.py has no load_video function to replace")
    current = text[start:end]
    if 'set_bridge("native")' in current and "import decord\\n" in text:
        if text != path.read_text(encoding="utf-8"):
            path.write_text(text, encoding="utf-8")
        drop_imported_vbench()
        return "Human_Clothes already resets Decord to native before asnumpy()"
    replacement = "\\n".join([
        "def load_video(video_path, max_frames_num, fps=1, force_sample=False):",
        "    if max_frames_num == 0:",
        "        return np.zeros((1, 336, 336, 3))",
        "    # Human Identity sets the torch bridge for the whole process.",
        "    # get_batch must return a Decord array so asnumpy() exists.",
        '    decord.bridge.set_bridge("native")',
        "    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)",
        "    total_frame_num = len(vr)",
        "    video_time = total_frame_num / vr.get_avg_fps()",
        "    fps = round(vr.get_avg_fps() / fps)",
        "    frame_idx = [i for i in range(0, len(vr), fps)]",
        "    frame_time = [i / fps for i in frame_idx]",
        "    if len(frame_idx) > max_frames_num or force_sample:",
        "        sample_fps = max_frames_num",
        "        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)",
        "        frame_idx = uniform_sampled_frames.tolist()",
        "        frame_time = [i / vr.get_avg_fps() for i in frame_idx]",
        '    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])',
        "    batch = vr.get_batch(frame_idx)",
        '    spare_frames = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.detach().cpu().numpy()',
        "    return spare_frames, frame_time, video_time",
        "",
        "",
    ])
    text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")
    drop_imported_vbench()
    return "Human_Clothes resets Decord to native before asnumpy()"


def keep_clothes_results_when_none_qualify(checkout: Path) -> str:
    # Every clip can be -1. VBench then divides the empty total by zero and
    # discards the per-video results. Keep those results and skip the average.
    path = checkout / "VBench-2.0" / "vbench2" / "human_clothes.py"
    text = path.read_text(encoding="utf-8")
    old_return = "    return final_score/valid_num, processed_json\\n"
    new_return = (
        "    if valid_num == 0:\\n"
        "        return None, processed_json\\n"
        "    return final_score/valid_num, processed_json\\n"
    )
    old_avg = "    all_results = score/num\\n"
    new_avg = (
        "    if num == 0:\\n"
        "        all_results = None\\n"
        "    else:\\n"
        "        all_results = score/num\\n"
    )
    if "if valid_num == 0:" in text and "if num == 0:" in text:
        drop_imported_vbench()
        return "Human_Clothes keeps per-video results when every clip is -1"
    if old_return not in text or old_avg not in text:
        raise RuntimeError("human_clothes.py has no empty-average division to guard")
    text = text.replace(old_return, new_return, 1).replace(old_avg, new_avg, 1)
    path.write_text(text, encoding="utf-8")
    drop_imported_vbench()
    return "Human_Clothes no longer divides by zero when every clip is -1"


def trace_llava_clothes(checkout: Path) -> str:
    # Print the chat prompt and the decoded answer inside the pinned Human_Clothes function.
    path = checkout / "VBench-2.0" / "vbench2" / "human_clothes.py"
    text = path.read_text(encoding="utf-8")
    if "Human_Clothes PROMPT:" in text and "Human_Clothes OUTPUT:" in text:
        drop_imported_vbench()
        return "Human_Clothes prints each prompt and decoded answer"
    replacements = [
        (
            "                prompt_question = conv.get_prompt()\\n"
            "                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=\\"pt\\").unsqueeze(0).to(device)\\n",
            "                prompt_question = conv.get_prompt()\\n"
            "                print(\\"\\\\nHuman_Clothes PROMPT:\\", os.path.basename(str(video_path)), f\\"question {i + 1}/{len(base_question)}\\", flush=True)\\n"
            "                print(prompt_question, flush=True)\\n"
            "                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=\\"pt\\").unsqueeze(0).to(device)\\n",
        ),
        (
            "                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()\\n",
            "                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()\\n"
            "                print(\\"Human_Clothes OUTPUT:\\", text_outputs, flush=True)\\n",
        ),
        (
            "            if not valid:\\n"
            "                new_item[f\\"video_results\\"]=-1\\n"
            "                processed_json.append(new_item)\\n",
            "            if not valid:\\n"
            "                new_item[f\\"video_results\\"]=-1\\n"
            "                print(\\"Human_Clothes RECORDED:\\", os.path.basename(str(video_path)), new_item[\\"video_results\\"], flush=True)\\n"
            "                processed_json.append(new_item)\\n",
        ),
        (
            "            valid_num+=1\\n"
            "\\n"
            "            processed_json.append(new_item)\\n",
            "            valid_num+=1\\n"
            "            print(\\"Human_Clothes RECORDED:\\", os.path.basename(str(video_path)), new_item[\\"video_results\\"], flush=True)\\n"
            "\\n"
            "            processed_json.append(new_item)\\n",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            raise RuntimeError("Human_Clothes trace patch did not match vbench2/human_clothes.py")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    drop_imported_vbench()
    return "Human_Clothes will print each LLaVA prompt and decoded answer"


def install_anatomy_stack():
    if not INSTALL_SCORING_DEPS:
        return
    import torch
    version = torch.__version__.split("+")[0]
    torch_tag = ".".join(version.split(".")[:2])
    cuda = getattr(torch.version, "cuda", None) or ""
    cu_tag = "cu" + "".join(cuda.split(".")[:2]) if cuda else ""
    index = f"https://download.openmmlab.com/mmcv/dist/{cu_tag}/torch{torch_tag}/index.html"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mmengine", "mmdet", "mmyolo"])
        if pip_would_change_torch(["mmcv==2.2.0"]):
            print("Skipping mmcv==2.2.0 because it would replace the CUDA torch build.")
            print("Human_Anatomy will be left missing if its import needs that wheel.")
            return
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "mmcv==2.2.0",
            "-f", index, "--no-cache-dir",
        ])
        print("Human_Anatomy packages installed from", index)
    except Exception as exc:
        print("Human_Anatomy packages were not installed.")
        print(exc)
        print("The other dimensions still run.")


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

Each target dimension is submitted alone with `mode='custom_input'`. Diversity and Multi-View_Consistency are not submitted. Set `RQ1_RUN_SCORING=0` to rebuild the metric table skeleton / reuse an existing file without calling VBench again.
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
    rejected = []
    for video_id, staged in staged_by_id.items():
        value = lookup_score(parsed, staged.name) if parsed else None
        if dimension in {"Human_Clothes", "Human_Identity"} and value is not None and float(value) < 0:
            rejected.append(video_id)
            value = None
        per_video[video_id] = value
        if value is None and video_id not in rejected:
            missing.append(video_id)
    reason = f"Read per-video custom_input scores from {result_path.name}."
    if rejected:
        if dimension == "Human_Identity":
            detail = "fewer than 20 frames contained exactly one face, so identity is not scored"
        else:
            detail = "the clip is not one person throughout, so clothes consistency is not scored"
        reason += (
            " Left missing for "
            + ", ".join(rejected)
            + f" because VBench-2.0 returned -1 ({detail})."
        )
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
        for entry in (str(checkout / "VBench-2.0"), str(checkout)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        print("LLaVA-NeXT import root", add_llava_import_root(checkout))
        os.environ["VBENCH_CACHE_DIR"] = str(CACHE_DIR / "weights")
        os.environ["VBENCH2_CACHE_DIR"] = str(CACHE_DIR / "weights")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print(shim_torchvision_write_video())
        install_core_packages()
        print(ensure_decord())
        print(ensure_huggingface_cli())
        print(prefetch_retinaface())
        print(use_pytorch_sdpa_for_llava(checkout))
        print(fix_clothes_decord_bridge(checkout))
        print(keep_clothes_results_when_none_qualify(checkout))
        print(trace_llava_clothes(checkout))
        install_anatomy_stack()
        full_vbench = checkout / "vbench" / "VBench_full_info.json"
        full_vbench2 = checkout / "VBench-2.0" / "vbench2" / "VBench2_full_info.json"
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
                reason = "Reused complete scores already stored in metric_scores.csv."
                negative = already < 0
                if spec["dimension"] in {"Human_Identity", "Human_Clothes"} and bool(negative.any()):
                    ids = metrics.loc[negative, "video_id"].astype(str).tolist()
                    metrics.loc[negative, spec["column"]] = np.nan
                    if spec["dimension"] == "Human_Identity":
                        detail = "fewer than 20 frames contained exactly one face, so identity is not scored"
                    else:
                        detail = "the clip is not one person throughout, so clothes consistency is not scored"
                    reason += (
                        " Left missing for "
                        + ", ".join(ids)
                        + f" because VBench-2.0 returned -1 ({detail})."
                    )
                    print(" ", spec["dimension"], "already scored;", ", ".join(ids), "left missing (-1)")
                else:
                    print(" ", spec["dimension"], "already scored, left unchanged")
                per_dimension[spec["dimension"]] = {"status": "scored", "reason": reason}
                continue
            if spec["dimension"] == "Human_Anatomy":
                try:
                    import mmcv  # noqa: F401
                except Exception as exc:
                    msg = (
                        type(exc).__name__
                        + ": "
                        + str(exc)
                        + ". No mmcv wheel matches this PyTorch, so Human_Anatomy was not scored."
                    )
                    per_dimension[spec["dimension"]] = {"status": "failed", "reason": msg}
                    print(" ", spec["dimension"], "FAILED:", msg)
                    continue
            try:
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
    cells: list[dict] = []
    cells.append(md("""
# RQ1: Video-quality metrics and human enjoyment

**Research question:** How strongly are scores on selected VBench and VBench-2.0 dimensions associated with human viewers' enjoyment ratings of AI-generated drama clips?

Primary analysis: one Spearman correlation per prespecified dimension against mean human enjoyment per clip, with percentile bootstrap 95% confidence intervals, two-sided Monte Carlo permutation tests, and Holm correction across all selected dimensions.

**Data inputs (produced outside this notebook):**

- `data/metric_scores.csv` — from `notebooks/vbench_score_clips.ipynb`. On Colab that notebook downloads `rq1_vbench_csvs.zip`; unzip it into this `data/` folder first.
- `data/human_ratings.csv` — hand-built from Google Docs (`video_id`, `rater_id`, `enjoyment`)

This notebook does not run VBench and does not invent enjoyment ratings. If either CSV is missing, stop after the load cell.

## Scope and assumptions
- Unit of analysis: clip, not individual rating.
- Mean enjoyment assumes approximately equally spaced scale points. A median sensitivity analysis is included.
- Inference assumes independent clips. Repeated `source_id` values stop confirmatory inference (replace provisional `unlabeled_pilot` with real drama IDs first).
- Clip bootstrap intervals condition on the observed viewer panel.
- Correlations are not causal effects. A nonsignificant result is not evidence of no association.

`DEMO_MODE = True` builds a labeled synthetic panel so the statistical code can be inspected without study files. Those prints are not findings.
"""))

    cells.append(code("""
from pathlib import Path
import json, os, platform, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, rankdata
import statsmodels
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from IPython.display import display

DEMO_MODE = False

REPO_ROOT = Path("..").resolve()
if not (REPO_ROOT / "data").exists() and (Path.cwd() / "data").exists():
    REPO_ROOT = Path.cwd().resolve()
DATA_DIR = Path(os.environ.get("RQ1_DATA_DIR", REPO_ROOT / "data")).resolve()
OUT_DIR = Path(os.environ.get(
    "RQ1_OUT_DIR",
    REPO_ROOT / ("rq1_outputs_demo" if DEMO_MODE else "rq1_outputs"),
)).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "vbench2_human_anatomy",
    "vbench2_human_identity",
    "vbench2_human_clothes",
]
SCALE_MIN, SCALE_MAX = 1, 5
MIN_RATERS_PER_CLIP = 3
ALPHA = 0.05
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 4999
SEED = 42

ANALYSIS_READY = False
print("DATA_DIR =", DATA_DIR)
print("OUT_DIR =", OUT_DIR)
"""))

    cells.append(md("""
## 1. Load data

Research mode reads `data/metric_scores.csv` and `data/human_ratings.csv`. Demo mode uses synthetic rows only.
"""))

    cells.append(code("""
rng = np.random.default_rng(SEED)
pd.DataFrame(columns=["video_id", "source_id"] + DIMENSIONS).to_csv(
    OUT_DIR / "metric_scores_template.csv", index=False
)
pd.DataFrame(columns=["video_id", "rater_id", "enjoyment"]).to_csv(
    OUT_DIR / "human_ratings_template.csv", index=False
)

metrics_path = DATA_DIR / "metric_scores.csv"
ratings_path = DATA_DIR / "human_ratings.csv"

if DEMO_MODE:
    n = 40
    ids = [f"clip_{i:03d}" for i in range(n)]
    metrics = pd.DataFrame({"video_id": ids, "source_id": [f"drama_{i:03d}" for i in range(n)]})
    for dimension in DIMENSIONS:
        metrics[dimension] = rng.uniform(0, 1, n)
    latent = (
        2
        + 1.5 * metrics[DIMENSIONS[0]].to_numpy()
        + 0.6 * metrics[DIMENSIONS[1]].to_numpy()
    )
    rows = []
    for j in range(12):
        bias = rng.normal(0, 0.3)
        for i, video_id in enumerate(ids):
            rating = int(np.clip(
                np.rint(latent[i] + bias + rng.normal(0, 0.8)),
                SCALE_MIN, SCALE_MAX,
            ))
            rows.append((video_id, f"viewer_{j:03d}", rating))
    ratings = pd.DataFrame(rows, columns=["video_id", "rater_id", "enjoyment"])
    ANALYSIS_READY = True
    print("SYNTHETIC DEMO - not research results")
else:
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run notebooks/vbench_score_clips.ipynb first."
        )
    metrics = pd.read_csv(metrics_path, dtype={"video_id": str, "source_id": str})
    if not ratings_path.is_file():
        ANALYSIS_READY = False
        ratings = pd.DataFrame(columns=["video_id", "rater_id", "enjoyment"])
        print(
            f"Missing {ratings_path}. "
            "Create it from Google Docs with columns video_id,rater_id,enjoyment "
            "(integers 1-5). Association cells will not run until that file exists."
        )
        display(metrics.head())
    else:
        ratings = pd.read_csv(ratings_path, dtype={"video_id": str, "rater_id": str})
        ANALYSIS_READY = True
        display(metrics.head(), ratings.head())
"""))

    cells.append(code("""
assert len(DIMENSIONS) > 0 and len(set(DIMENSIONS)) == len(DIMENSIONS)
assert {"video_id", *DIMENSIONS}.issubset(metrics.columns), "Missing metric columns"

if not ANALYSIS_READY:
    analysis = pd.DataFrame()
    print("Aggregation skipped until human_ratings.csv exists.")
else:
    assert {"video_id", "rater_id", "enjoyment"}.issubset(ratings.columns), "Missing rating columns"
    assert not metrics["video_id"].isna().any() and not ratings[["video_id", "rater_id"]].isna().any().any()
    assert not metrics["video_id"].duplicated().any(), "Metric table must have one row per clip"
    assert not ratings.duplicated(["video_id", "rater_id"]).any(), "Resolve duplicate viewer-clip records"
    for col in DIMENSIONS:
        metrics[col] = pd.to_numeric(metrics[col], errors="raise")
    ratings["enjoyment"] = pd.to_numeric(ratings["enjoyment"], errors="raise")
    assert not np.isinf(metrics[DIMENSIONS].to_numpy(dtype=float)).any()
    assert not np.isinf(ratings["enjoyment"].to_numpy(dtype=float)).any()
    observed = ratings["enjoyment"].dropna()
    assert observed.between(SCALE_MIN, SCALE_MAX).all(), "Ratings outside configured scale"
    unknown = set(ratings["video_id"]) - set(metrics["video_id"])
    assert not unknown, f"Ratings without metric-table records: {unknown}"
    print("Missing enjoyment rows excluded:", int(ratings["enjoyment"].isna().sum()))
    valid_ratings = ratings.dropna(subset=["enjoyment"]).copy()
    summary = valid_ratings.groupby("video_id")["enjoyment"].agg(
        human_mean="mean", human_median="median", n_raters="count", human_sd="std"
    ).reset_index()
    clips = metrics.merge(summary, on="video_id", how="left", validate="one_to_one")
    clips["included"] = clips["n_raters"].fillna(0).ge(MIN_RATERS_PER_CLIP)
    clips.to_csv(OUT_DIR / "clip_summary_and_exclusions.csv", index=False)
    print("Clips excluded for insufficient ratings:", int((~clips["included"]).sum()))
    analysis = clips.loc[clips["included"]].copy()
    assert len(analysis) >= 4, "At least four eligible clips required; substantially more are recommended"
    if "source_id" in analysis:
        assert not analysis["source_id"].isna().any(), "Fill source IDs or remove the optional column"
        assert not analysis["source_id"].duplicated().any(), (
            "Repeated drama/source IDs: independent-clip inference is inappropriate. "
            "Replace provisional unlabeled_pilot values with real drama IDs "
            "(one clip per independent source), or use a cluster-aware design."
        )
    else:
        warnings.warn("No source_id supplied: verify clip independence manually.")
    display(analysis[["n_raters", "human_mean", "human_sd"]].describe())
"""))

    cells.append(md("""
## 2. Spearman correlations and uncertainty

Permutation tests shuffle clip-level enjoyment relative to metric scores. The Monte Carlo p-value uses the absolute correlation and a plus-one correction. Bootstrap intervals resample paired clips and recalculate ranks; constant resamples are discarded and counted. These are marginal, not simultaneous, 95% intervals. Missing metric values are handled separately per dimension, with sample sizes reported. Holm correction includes all prespecified dimensions; untestable dimensions enter the correction conservatively as p = 1 and retain missing reported results.

SciPy recommends considering permutation tests rather than relying on the asymptotic Spearman p-value for small samples: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

Holm correction documentation: https://www.statsmodels.org/devel/generated/statsmodels.stats.multitest.multipletests.html
"""))

    cells.append(code("""
def association(x, y, seed):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    result = dict(
        n_clips=len(x), rho=np.nan, ci_low=np.nan, ci_high=np.nan,
        p_permutation=np.nan, valid_bootstraps=0, status="untestable",
    )
    if len(x) < 4 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return result
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    rho = float(rx @ ry / denominator)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        perm_rho = rx @ rng.permutation(ry) / denominator
        extreme += abs(perm_rho) >= abs(rho) - 1e-12
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            boot.append(float(spearmanr(x[idx], y[idx]).statistic))
    low, high = (
        np.quantile(boot, [0.025, 0.975]) if len(boot) >= 100 else (np.nan, np.nan)
    )
    result.update(
        rho=rho, ci_low=low, ci_high=high,
        p_permutation=(extreme + 1) / (N_PERMUTATIONS + 1),
        valid_bootstraps=len(boot), status="tested",
    )
    return result

if ANALYSIS_READY:
    results = pd.DataFrame([
        dict(dimension=d, **association(analysis[d], analysis["human_mean"], SEED + i))
        for i, d in enumerate(DIMENSIONS)
    ])
    testable = results["status"].eq("tested")
    reject, adjusted, _, _ = multipletests(
        results["p_permutation"].fillna(1), alpha=ALPHA, method="holm"
    )
    results["p_holm"] = np.where(testable, adjusted, np.nan)
    results["reject_holm"] = testable & reject
    results["abs_rho"] = results["rho"].abs()
    results = results.sort_values("abs_rho", ascending=False, na_position="last")
    results.to_csv(OUT_DIR / "rq1_spearman_results.csv", index=False)
    display(results.drop(columns="abs_rho").round(4))
else:
    results = pd.DataFrame()
    print("Spearman cells skipped: waiting for data/human_ratings.csv")
"""))

    cells.append(md("""
## 3. Plots and median sensitivity analysis

Sensitivity correlations are descriptive checks, not a second set of confirmatory tests.
"""))

    cells.append(code("""
if not ANALYSIS_READY:
    print("Plots skipped: waiting for data/human_ratings.csv")
else:
    plot_data = results.dropna(subset=["rho"]).iloc[::-1]
    if len(plot_data):
        fig, ax = plt.subplots(figsize=(9, max(3, len(plot_data) * 0.5)))
        for i, row in enumerate(plot_data.itertuples()):
            color = "tab:blue" if row.reject_holm else "gray"
            ax.plot([row.ci_low, row.ci_high], [i, i], color=color)
            ax.scatter(row.rho, i, color=color)
        ax.set_yticks(range(len(plot_data)), plot_data["dimension"])
        ax.axvline(0, color="black", lw=0.8)
        ax.set(
            xlim=(-1.05, 1.05),
            xlabel="Spearman rho (marginal bootstrap 95% CI)",
            title=("DEMO: " if DEMO_MODE else "") + "Metric-enjoyment associations",
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / "rq1_correlations.png", dpi=180, bbox_inches="tight")
        plt.show()

    sensitivity = []
    for d in DIMENSIONS:
        subset = analysis[[d, "human_median"]].dropna()
        rho = (
            spearmanr(subset[d], subset["human_median"]).statistic
            if (
                len(subset) >= 4
                and subset[d].nunique() > 1
                and subset["human_median"].nunique() > 1
            )
            else np.nan
        )
        sensitivity.append({"dimension": d, "n_clips": len(subset), "rho_human_median": rho})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT_DIR / "median_sensitivity.csv", index=False)
    display(sensitivity)

    fig, axes = plt.subplots(len(DIMENSIONS), 1, figsize=(7, 3 * len(DIMENSIONS)), squeeze=False)
    for ax, d in zip(axes.flat, DIMENSIONS):
        ax.scatter(analysis[d], analysis["human_mean"], alpha=0.7)
        ax.set(xlabel=d, ylabel="Mean human enjoyment", ylim=(SCALE_MIN - 0.1, SCALE_MAX + 0.1))
    fig.suptitle("SYNTHETIC DEMO" if DEMO_MODE else "Clip-level observations")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rq1_scatterplots.png", dpi=180, bbox_inches="tight")
    plt.show()
"""))

    cells.append(code("""
metadata = {
    "demo_mode": DEMO_MODE,
    "analysis_ready": ANALYSIS_READY,
    "dimensions": DIMENSIONS,
    "scale": [SCALE_MIN, SCALE_MAX],
    "minimum_raters": MIN_RATERS_PER_CLIP,
    "alpha": ALPHA,
    "bootstrap_resamples": N_BOOTSTRAP,
    "permutations": N_PERMUTATIONS,
    "seed": SEED,
    "eligible_clips": int(len(analysis)) if ANALYSIS_READY else 0,
    "data_dir": str(DATA_DIR),
    "metrics_csv": str(metrics_path if not DEMO_MODE else "synthetic"),
    "ratings_csv": str(ratings_path if not DEMO_MODE else "synthetic"),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
}
(OUT_DIR / "analysis_settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if not ANALYSIS_READY:
    report = (
        "RQ1 association waiting for data/human_ratings.csv.\\n"
        "metric_scores.csv is loaded from the VBench scoring notebook when present.\\n"
        "Add Google Docs ratings manually, then rerun this notebook."
    )
else:
    lines = [
        "SYNTHETIC DEMO - NOT STUDY FINDINGS" if DEMO_MODE else "RQ1 analysis summary",
        f"Eligible clips: {len(analysis)}; mean enjoyment aggregated per clip.",
        "Two-sided clip permutation p-values; Holm correction across selected dimensions.",
        "Intervals are marginal paired-clip bootstrap intervals, conditional on the observed viewer panel.",
    ]
    for row in results.itertuples():
        if row.status == "tested":
            lines.append(
                f"{row.dimension}: rho={row.rho:.3f}, 95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}], "
                f"Holm p={row.p_holm:.4f}, n={row.n_clips}."
            )
        else:
            lines.append(f"{row.dimension}: untestable (insufficient data or no variation).")
    lines += [
        "Associations are not causal effects. Nonsignificance is not evidence of no association.",
        "Largest absolute sample correlation does not prove a dimension is stronger than all others.",
    ]
    report = "\\n".join(lines)

(OUT_DIR / "rq1_summary.txt").write_text(report, encoding="utf-8")
print(report)
"""))

    cells.append(md("""
## Reporting checklist

Report clip and viewer counts, source sampling, scale anchors, metric versions/configurations, chosen dimensions and rationale, missingness/exclusions, per-dimension sample sizes, rho and confidence intervals, Holm-adjusted p-values, and limitations of clip independence and viewer sampling.

Example sentence (replace placeholders; never copy demo estimates):

> Across N eligible clips, dimension X was positively associated with mean viewer enjoyment (Spearman rho = R, marginal bootstrap 95% CI [L, U], Holm-adjusted permutation p = P). This association does not establish causation or an independent contribution beyond other dimensions.

Scoring: `notebooks/vbench_score_clips.ipynb`. Ratings: manual `data/human_ratings.csv` from Google Docs.
"""))

    write_nb(NB_DIR / "rq1-aidrama.ipynb", cells)


if __name__ == "__main__":
    build_scoring_notebook()
    build_association_notebook()
