"""Run the environment check directly from a source checkout, without installing it."""

from pathlib import Path
import sys

# Make this entry point independent of the caller's current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_attendance.environment import main


if __name__ == "__main__":
    raise SystemExit(main())
