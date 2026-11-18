"""Download and version the Paddy Doctor competition data with DVC."""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from dvc.exceptions import DvcException
from dvc.repo import Repo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "paddy-disease-classification"
DATA_PATHS = ("train.csv", "sample_submission.csv", "train_images", "test_images")
DATA_TARGETS = {
    "train": ("train.csv", "train_images"),
    "infer": ("test_images", "sample_submission.csv"),
}


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract an archive downloaded from Kaggle into a temporary directory."""
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            output = (destination / member.filename).resolve()
            if not output.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe path in Kaggle archive: {member.filename}")
        bundle.extractall(destination)


def _validate_data(root: Path) -> None:
    """Check that the extracted dataset has the expected competition structure."""
    with (root / "train.csv").open(newline="") as input_file:
        train_rows = list(csv.DictReader(input_file))
    with (root / "sample_submission.csv").open(newline="") as input_file:
        test_rows = list(csv.DictReader(input_file))

    if len(train_rows) != 10407 or len(test_rows) != 3469:
        raise ValueError("Unexpected number of rows in the Kaggle CSV files")
    if len({row["label"] for row in train_rows}) != 10:
        raise ValueError("Expected ten classes in train.csv")
    if any(
        not (root / "train_images" / row["label"] / row["image_id"]).is_file()
        for row in train_rows
    ):
        raise ValueError("Some labeled training images are missing")
    if any(not (root / "test_images" / row["image_id"]).is_file() for row in test_rows):
        raise ValueError("Some test images are missing")
    if sum(1 for _ in (root / "train_images").rglob("*.jpg")) != len(train_rows):
        raise ValueError("Unexpected number of training images")
    if sum(1 for _ in (root / "test_images").glob("*.jpg")) != len(test_rows):
        raise ValueError("Unexpected number of test images")


def download_data(root: Path = PROJECT_ROOT, archive: Path | None = None) -> Path:
    """Get data from Kaggle and fill the local DVC cache and remote.

    If the four data paths already exist, only validation and DVC bootstrap run.
    Otherwise, use a previously downloaded archive or the Kaggle CLI. Kaggle
    authentication and acceptance of the competition rules may be required.
    """
    root = Path(root).resolve()
    missing = [name for name in DATA_PATHS if not (root / name).exists()]
    if missing:
        with tempfile.TemporaryDirectory(prefix="paddy_kaggle_") as temp_name:
            temp_dir = Path(temp_name)
            if archive is None:
                kaggle_cli = shutil.which("kaggle")
                if kaggle_cli is None:
                    raise RuntimeError("Kaggle CLI is unavailable; run with uv")
                subprocess.run(
                    [
                        kaggle_cli,
                        "competitions",
                        "download",
                        COMPETITION,
                        "--path",
                        str(temp_dir),
                        "--unzip",
                    ],
                    check=True,
                )
            else:
                _extract_archive(Path(archive).resolve(), temp_dir)

            for nested_archive in list(temp_dir.rglob("*.zip")):
                _extract_archive(nested_archive, nested_archive.parent)

            for name in missing:
                matches = list(temp_dir.rglob(name))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one {name!r} in Kaggle download; "
                        f"found {len(matches)}"
                    )
                shutil.move(str(matches[0]), str(root / name))

    _validate_data(root)

    metadata = {name: (root / f"{name}.dvc").read_bytes() for name in DATA_PATHS}
    with Repo(str(root)) as repo:
        repo.add(DATA_PATHS)
        changed = [
            name
            for name in DATA_PATHS
            if (root / f"{name}.dvc").read_bytes() != metadata[name]
        ]
        if changed:
            for name in changed:
                (root / f"{name}.dvc").write_bytes(metadata[name])
            raise ValueError(
                "Downloaded files differ from the DVC-pinned dataset: "
                + ", ".join(changed)
            )
        repo.push()
    return root


def ensure_data(stage: str, root: Path = PROJECT_ROOT) -> Path:
    """Pull DVC data for training or inference, bootstrapping from Kaggle if needed."""
    if stage not in DATA_TARGETS:
        raise ValueError(f"Unknown data stage: {stage!r}")

    root = Path(root).resolve()
    targets = [f"{name}.dvc" for name in DATA_TARGETS[stage]]
    try:
        with Repo(str(root)) as repo:
            repo.pull(targets=targets)
    except DvcException:
        download_data(root)
        with Repo(str(root)) as repo:
            repo.pull(targets=targets)

    missing = [name for name in DATA_TARGETS[stage] if not (root / name).exists()]
    if missing:
        raise FileNotFoundError("DVC did not restore: " + ", ".join(missing))
    return root
