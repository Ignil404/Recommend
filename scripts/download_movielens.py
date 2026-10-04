"""
Download and verify MovieLens 1M dataset.

Downloads from GroupLens, verifies checksums, and extracts to data/ml-1m/.
Supports resume and skip-if-verified.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Official MovieLens 1M dataset info
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_FILES: dict[str, tuple[int, str]] = {
    "ml-1m/ratings.dat": (24594131, "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f"),
    "ml-1m/users.dat": (134368, "1dc3a95300cb19f7c10e027daf29b8cdf1d908dab65b5edfb45950aa389a10a3"),
    "ml-1m/movies.dat": (171308, "0140fc2356357c1a851d0f52e893a1e4d3696df632c4141cea8d5bc3d621f0b9"),
}

DATA_DIR = Path("data")
ML1M_DIR = DATA_DIR / "ml-1m"
ZIP_PATH = DATA_DIR / "ml-1m.zip"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_files() -> bool:
    """Verify all expected files exist with correct size and checksum."""
    for rel_path, (expected_size, expected_hash) in EXPECTED_FILES.items():
        filepath = DATA_DIR / rel_path
        if not filepath.exists():
            return False
        if filepath.stat().st_size != expected_size:
            print(f"Size mismatch for {rel_path}: {filepath.stat().st_size} != {expected_size}")
            return False
        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            print(f"Checksum mismatch for {rel_path}: {actual_hash} != {expected_hash}")
            return False
    return True


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress bar, supporting resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"

    response = requests.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0)) + dest.stat().st_size if dest.exists() else int(response.headers.get("Content-Length", 0))
    mode = "ab" if dest.exists() else "wb"

    with open(dest, mode) as f, tqdm(
        total=total_size,
        initial=dest.stat().st_size if dest.exists() else 0,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file to destination."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def main() -> int:
    """Main entry point."""
    print("MovieLens 1M Dataset Downloader")
    print("=" * 40)

    # Check if already verified
    if verify_files():
        print("✓ All files already present and verified")
        return 0

    # Download zip
    print(f"Downloading from {ML1M_URL}...")
    try:
        download_with_progress(ML1M_URL, ZIP_PATH)
    except (requests.RequestException, OSError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    # Extract
    print("Extracting...")
    try:
        extract_zip(ZIP_PATH, DATA_DIR)
    except (zipfile.BadZipFile, OSError) as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        return 1

    # Verify
    print("Verifying...")
    if verify_files():
        print("✓ Download and verification successful")
        # Clean up zip
        ZIP_PATH.unlink(missing_ok=True)
        return 0
    else:
        print("✗ Verification failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())