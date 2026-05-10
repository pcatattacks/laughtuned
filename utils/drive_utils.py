"""Google Drive helpers for the Colab runtime.

Mounts Drive and pre-creates the artifact directory tree expected by the
rest of the pipeline. All artifact paths are derived from
``CONFIG["drive_root"]`` so a different mount point only needs to change
``config.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


# Subdirectories created under ``drive_root`` on first run.
# Mirrors the tree specified in ``instructions.md``.
_ARTIFACT_SUBDIRS: List[str] = [
    "data/articles",
    "data/prompts",
    "data/generations",
    "data/preferences",
    "data/splits",
    "data/eval",
    "checkpoints",
    "metrics",
    "figures",
    "ref_log_probs",
]


def mount_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive in the current Colab runtime.

    A no-op when not running on Colab so the same notebook can be opened
    locally without erroring out.

    Args:
        mount_point: Filesystem path where Drive will be mounted.
            Defaults to Colab's conventional ``/content/drive``.
    """
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        print("[drive_utils] Not running on Colab; skipping Drive mount.")
        return

    drive.mount(mount_point)
    print(f"[drive_utils] Mounted Drive at {mount_point}")


def ensure_drive_dirs(config: Dict[str, Any]) -> None:
    """Create the LaughTuned artifact directory tree on Drive.

    Idempotent: existing directories are left as-is.

    Args:
        config: Project config dict. Must contain ``drive_root``.
    """
    drive_root: str = config["drive_root"]
    if not os.path.isdir(os.path.dirname(drive_root)):
        print(
            f"[drive_utils] Parent of drive_root does not exist: "
            f"{os.path.dirname(drive_root)!r}. "
            "Did you mount Drive first?"
        )
        return

    os.makedirs(drive_root, exist_ok=True)
    for subdir in _ARTIFACT_SUBDIRS:
        os.makedirs(os.path.join(drive_root, subdir), exist_ok=True)

    print(
        f"[drive_utils] Ready: {drive_root} "
        f"({len(_ARTIFACT_SUBDIRS)} subdirectories ensured)"
    )
