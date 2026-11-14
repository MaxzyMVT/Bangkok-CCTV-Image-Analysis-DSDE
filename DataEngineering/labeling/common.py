"""Shared paths and constants for the test-set labeling workflow."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_CSV = ROOT_DIR / "DataEngineering" / "cctv_images_metadata.csv"
TEST_SET_DIR = ROOT_DIR / "DataEngineering" / "test_set"
WORK_DIR = ROOT_DIR / "labeling_work"
DEFAULT_DATASET_ROOT = ROOT_DIR  # folder that contains dataset/100_cameras/...

CANDIDATES_CSV = TEST_SET_DIR / "candidates.csv"
ASSIGNMENT_CSV = TEST_SET_DIR / "camera_assignment.csv"

# Target classes (same mapping as Models/training_prep.py)
TARGET_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
COCO_TO_TARGET = {2: 0, 3: 1, 5: 2, 7: 3}

# BGR colors for review previews
CLASS_COLORS = {
    0: (0, 165, 255),    # car - orange
    1: (255, 80, 200),   # motorcycle - pink
    2: (60, 220, 60),    # bus - green
    3: (255, 200, 0),    # truck - cyan/blue
}
CLASS_LETTER = {0: "C", 1: "M", 2: "B", 3: "T"}


def resolve_dataset_root(arg: str | None) -> Path:
    """Folder that holds dataset/100_cameras. Falls back to the main checkout when run from a worktree."""
    if arg:
        return Path(arg).resolve()
    if (DEFAULT_DATASET_ROOT / "dataset" / "100_cameras").exists():
        return DEFAULT_DATASET_ROOT
    # .claude/worktrees/<name>/ -> main checkout is three levels up
    for parent in ROOT_DIR.parents:
        if (parent / "dataset" / "100_cameras").exists():
            return parent
    raise FileNotFoundError(
        "Cannot find dataset/100_cameras. Pass --dataset-root <folder containing dataset/>."
    )


def find_image(root: Path, relative_path: str) -> Path:
    """Resolve an image path from the metadata CSV, accepting both dataset layouts:
    dataset/100_cameras/cctv_images/<cam>/<file>  (as in the metadata) and
    dataset/cctv_images/<cam>/<file>              (zip extracted one level up)."""
    p = root / relative_path
    if p.exists():
        return p
    alt = root / relative_path.replace("dataset/100_cameras/", "dataset/", 1)
    return alt if alt.exists() else p


def preview_name(camera_id, image_id) -> str:
    """Short ASCII file stem used for every per-image file in the workflow."""
    return f"cam{int(camera_id):04d}_img{int(image_id):05d}"
