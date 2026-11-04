"""Acquire the assigned competition through the pinned KaggleHub API."""

import os
from pathlib import Path

from .data_loader import FaceLayout


COMPETITION = "11-785-fall-20-homework-2-part-2"


def download_face_data(*, cache_dir: Path) -> FaceLayout:
    """Return the discovered extracted dataset root without copying raw images.

    Keep KaggleHub's reusable cache in the requested local directory, call the
    assigned competition API, then use its returned path rather than assuming
    a cache layout. Modern API tokens are managed by KaggleHub outside source
    code; this function never reads or logs them.
    """
    import kagglehub

    # The caller owns cache location (local disk or a Colab mount), while
    # KaggleHub owns the internal directory layout and token authentication.
    previous = os.environ.get("KAGGLEHUB_CACHE")
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir.expanduser().resolve())
    try:
        downloaded = Path(kagglehub.competition_download(COMPETITION))
    finally:
        # Restore process state even on authentication/network failure so another
        # download in the same notebook is not redirected into this project.
        if previous is None:
            os.environ.pop("KAGGLEHUB_CACHE", None)
        else:
            os.environ["KAGGLEHUB_CACHE"] = previous
    return FaceLayout.discover(downloaded)
