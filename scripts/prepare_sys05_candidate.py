"""Create an identified, unstaged local review snapshot under artifacts/.

Reads Git HEAD and overlays an explicit SYS-05 allowlist. Never stages, commits,
pushes, merges or publishes; the snapshot is not an independently frozen release.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = "reports/environment/vy-mac-m1-review.json"
DOCS = ("docs/sys-05-phase-1.md", "docs/sys-05-legacy-39ac3df.md", "docs/sys-05-owner-self-review.md",
        "docs/sys-05-contract-traceability.md", "docs/sys-05-handover.md")
SCRIPTS = ("scripts/setup_sys05_yunet.py", "scripts/benchmark_sys05.py", "scripts/smoke_sys05_gui.py",
           "scripts/demo_sys05.py", "scripts/prepare_sys05_candidate.py")


def prepare(destination, with_model=False):
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT / "artifacts") or destination == ROOT / "artifacts":
        raise ValueError("Candidate destination must be a new directory below this repository's artifacts/")
    paths = [*DOCS, *SCRIPTS, "requirements-sys05-haar.txt", "requirements-sys05-gui.txt"]
    paths += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "src/facial_attendance/ui").glob("*.py"))]
    paths += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "tests").glob("test_webcam*.py"))]
    for path in paths:
        if not (ROOT / path).is_file():
            raise ValueError(f"Required candidate file is missing: {path}")
    runtime_hashes = {path: sha256((ROOT / path).read_bytes()).hexdigest()
                      for path in paths if path.startswith("src/facial_attendance/ui/")}
    runtime_digest = sha256(json.dumps(runtime_hashes, sort_keys=True).encode()).hexdigest()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, members=[entry for entry in source.getmembers() if entry.name != PROTECTED], filter="data")
    evidence = ROOT / "reports/sys05/2026-11-16"
    if evidence.is_dir():
        paths += [str(path.relative_to(ROOT)) for path in sorted(evidence.iterdir())
                  if path.is_file() and path.name != "candidate-manifest.json"]
    for path in paths:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    if with_model:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from setup_sys05_yunet import FILES
        for name, _, digest, size in FILES:
            source = ROOT / "artifacts/sys05-yunet" / name
            if source.stat().st_size != size or sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Refusing unverified detector asset: {source}")
            target = destination / "artifacts/sys05-yunet" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            paths.append(str(target.relative_to(destination)))
    validation = evidence / "owner-physical-validation.json"
    owner = json.loads(validation.read_text()) if validation.is_file() else {}
    manifest = {
        "candidate_id": "SYS05-LOCAL-20261116-" + runtime_digest[:12],
        "created_utc": datetime.now(timezone.utc).isoformat(), "base_head": head,
        "runtime_digest": runtime_digest, "runtime_files_sha256": runtime_hashes,
        "owner_validation_matches_runtime": owner.get("runtime_digest") == runtime_digest,
        "status": "Local review candidate; no new commit, independent review or release freeze claimed",
        "excluded": [PROTECTED, ".git", "virtual environments", "datasets", "personal attendance/enrollment data"],
        "copied_files_sha256": {path: sha256((destination / path).read_bytes()).hexdigest() for path in sorted(paths)},
    }
    (destination / "SYS05-CANDIDATE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--with-model", action="store_true", help="Explicitly copy the already verified local model and licence")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.destination, args.with_model)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Candidate preparation failed: {error}\n")
    print(json.dumps({"candidate_id": result["candidate_id"], "destination": str(args.destination.resolve()),
                      "owner_validation_matches_runtime": result["owner_validation_matches_runtime"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
