"""
Bangkok CCTV Dataset Auto-Downloader
Downloads and extracts the official BMA CCTV dataset from the instructor's Google Drive.
Requires zero login / credentials.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import Optional

# Official Instructor Google Drive Assets
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1k2-KxXe4wbDmFGcGXI-pK7TdSXjBHTni?usp=sharing"
GDRIVE_IMAGES_ZIP_ID = "1mHP-N0qtbBy5IxhjZVovP_8g1Hw2H7dO"
GDRIVE_CCTV_INFO_ID = "1VSCzWeNOV7ds_d9BNJ9i20_9JzyncL2F"

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = ROOT_DIR / "dataset" / "100_cameras"
IMAGES_DIR = DEFAULT_TARGET_DIR / "cctv_images"
INFO_CSV_PATH = DEFAULT_TARGET_DIR / "cctv_info.csv"


def is_dataset_present(target_dir: Optional[Path] = None) -> bool:
    """Checks whether the CCTV image dataset already exists with actual files."""
    base = target_dir or DEFAULT_TARGET_DIR
    cctv_dir = base / "cctv_images"
    if not cctv_dir.exists():
        return False

    # Check for at least 5 camera folders with images
    cam_folders = [d for d in cctv_dir.iterdir() if d.is_dir()]
    if len(cam_folders) < 5:
        return False

    sample_has_images = any(any(d.glob("*.jpg")) for d in cam_folders[:5])
    return sample_has_images


def ensure_gdown():
    """Ensures gdown package is installed in the current Python environment."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("[Dataset Setup] 'gdown' package not found. Installing automatically via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        return gdown


def extract_zip_with_progress(zip_path: Path, extract_to: Path):
    """Extracts zip archive with progress reporting and path structure normalization."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting {zip_path.name} to {extract_to}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)

        # Inspect prefix to handle nested folders properly
        first_few = [m.filename for m in members[:10]]
        has_cctv_images_prefix = any(f.startswith("cctv_images/") or f.startswith("cctv_images\\") for f in first_few)
        has_100_cameras_prefix = any(f.startswith("100_cameras/") or f.startswith("100_cameras\\") for f in first_few)

        # Determine target root for extraction
        if has_100_cameras_prefix:
            dest_dir = extract_to.parent # dataset/
        elif has_cctv_images_prefix:
            dest_dir = extract_to # dataset/100_cameras/
        else:
            # If the zip directly contains camera folders like "1028/", extract to cctv_images/
            dest_dir = extract_to / "cctv_images"
            dest_dir.mkdir(parents=True, exist_ok=True)

        step = max(1, total_files // 20)
        for i, member in enumerate(members):
            zf.extract(member, dest_dir)
            if i % step == 0 or i == total_files - 1:
                pct = int((i + 1) / total_files * 100)
                sys.stdout.write(f"\r[Extracting] {pct}% completed ({i+1}/{total_files} files)")
                sys.stdout.flush()

    print("\n[✓] Extraction complete!")


def download_dataset_from_gdrive(target_dir: Optional[Path] = None, force: bool = False) -> bool:
    """
    Downloads cctv_info.csv and cctv_images.zip from instructor's Google Drive link
    and extracts them into target_dir (default: dataset/100_cameras).
    """
    base = target_dir or DEFAULT_TARGET_DIR
    cctv_dir = base / "cctv_images"
    info_csv = base / "cctv_info.csv"

    if not force and is_dataset_present(base):
        print(f"[✓] CCTV dataset already present at {cctv_dir}")
        return True

    print("\n" + "=" * 70)
    print(" [BMA CCTV Dataset Setup]")
    print(f" Dataset missing from: {cctv_dir}")
    print(f" Downloading from Google Drive: {GDRIVE_FOLDER_URL}")
    print("=" * 70 + "\n")

    gdown_module = ensure_gdown()
    base.mkdir(parents=True, exist_ok=True)

    # 1. Download cctv_info.csv if missing
    if not info_csv.exists() or force:
        print("[*] Downloading camera metadata (cctv_info.csv)...")
        try:
            gdown_module.download(
                id=GDRIVE_CCTV_INFO_ID,
                output=str(info_csv),
                quiet=False
            )
            print(f"[✓] Saved metadata to {info_csv}")
        except Exception as e:
            print(f"[!] Warning: Failed downloading cctv_info.csv via gdown: {e}")

    # 2. Download cctv_images.zip
    temp_zip = base.parent / "cctv_images.zip"
    print(f"[*] Downloading CCTV images archive (cctv_images.zip, ~2.5 GB)...")
    print(f"[*] Download destination: {temp_zip}")

    try:
        url = f"https://drive.google.com/uc?id={GDRIVE_IMAGES_ZIP_ID}"
        gdown_module.download(
            url=url,
            output=str(temp_zip),
            quiet=False,
            fuzzy=True
        )
    except Exception as e:
        print(f"\n[!] Error downloading cctv_images.zip: {e}")
        print("\n" + "-" * 70)
        print(" [Fallback Instructions for Teammates]")
        print(" If Google Drive shows 'Download quota exceeded':")
        print(f" 1. Open Google Drive: {GDRIVE_FOLDER_URL}")
        print(" 2. Download 'cctv_images.zip' and 'cctv_info.csv' manually.")
        print(f" 3. Place them inside: {base}")
        print(" 4. Re-run this command.")
        print("-" * 70 + "\n")
        return False

    if not temp_zip.exists() or temp_zip.stat().st_size < 10000:
        print(f"[!] Error: Downloaded file {temp_zip} is invalid or incomplete.")
        return False

    # 3. Extract Archive
    try:
        extract_zip_with_progress(temp_zip, base)
    except Exception as e:
        print(f"[!] Error extracting {temp_zip}: {e}")
        return False
    finally:
        # 4. Cleanup temporary archive to free disk space
        if temp_zip.exists():
            print(f"[*] Cleaning up temporary archive {temp_zip.name}...")
            try:
                temp_zip.unlink()
                print("[✓] Temporary zip file removed.")
            except Exception as e:
                print(f"[Warning] Could not remove temporary zip: {e}")

    # Verify
    if is_dataset_present(base):
        print("\n[✓] CCTV Dataset successfully downloaded and ready for use!\n")
        return True
    else:
        print("\n[!] Warning: Dataset extracted but directory verification failed.")
        return False


def ensure_dataset(target_dir: Optional[Path] = None) -> bool:
    """Convenience helper called at the start of app.py or CLI scripts."""
    base = target_dir or DEFAULT_TARGET_DIR
    if not is_dataset_present(base):
        print(f"[App Startup] Dataset not found in {base}. Launching auto-downloader...")
        return download_dataset_from_gdrive(base, force=False)
    return True


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    success = download_dataset_from_gdrive(force=force_download)
    sys.exit(0 if success else 1)
