"""Read dataset files with consistent path, JSON and checksum validation."""

import json
import shutil
from hashlib import file_digest
from pathlib import Path, PurePosixPath


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def local_path(root: Path, value: str) -> Path:
    """Resolve symbolic links and reject paths outside the dataset root."""
    relative = PurePosixPath(value.replace("\\", "/"))
    if not value or relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError(f"Unsafe dataset path: {value!r}")

    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside dataset: {value}")

    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value

    return result


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object
    )


def source_transitions(project=None):
    """Read explicit reviewed source moves from a local or staged release.

    Source compatibility is an exact before/after allowlist, never a relaxation
    of dataset, checkpoint or teacher checksums. Fresh runs need no transition.
    """
    project = (
        Path(project) if project is not None else Path(__file__).resolve().parents[3]
    )
    transitions = []
    for relative in (
        "configs/liveness-release.json",
        "transfer/release-inventory.json",
    ):
        path = project / relative
        if path.is_file():
            transitions.extend(read_json(path).get("source_transitions", []))
    return transitions


def compatible_sources(previous, current, project=None):
    if previous == current:
        return True
    return any(
        previous == scope["before"] and current == scope["after"]
        for transition in source_transitions(project)
        for scope in transition["scopes"]
    )


def implementation_matches(path, expected, project=None):
    """Accept a moved implementation only at its approved replacement checksum."""
    path = Path(path)
    if path.is_file() and file_sha256(path) == expected:
        return True
    project = (
        Path(project).resolve()
        if project is not None
        else Path(__file__).resolve().parents[3]
    )
    try:
        relative = path.resolve().relative_to(project).as_posix()
    except ValueError:
        return False
    for transition in source_transitions(project):
        if transition["before_files"].get(relative) != expected:
            continue
        target = transition.get("relocations", {}).get(relative, relative)
        replacement = (project / target).resolve()
        digest = transition["after_files"].get(target)
        if replacement.is_relative_to(project) and digest and replacement.is_file():
            units = transition.get("source_units", {}).get(relative, [target])
            # A split module is compatible only while every extracted implementation
            # still matches its reviewed checksum; checking its facade is insufficient.
            if units and all(
                (project / unit).resolve().is_relative_to(project)
                and (project / unit).is_file()
                and file_sha256(project / unit) == transition["after_files"].get(unit)
                for unit in units
            ):
                return True
    return False


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_root():
    """Return the liveness package root, including inside a producing snapshot."""
    return Path(__file__).resolve().parent


def source_path(relative):
    """Resolve a package-relative source file and reject missing/escaped paths."""
    return local_path(source_root(), relative)


def source_hashes():
    """Bind every Python implementation to its relative path, preserving subpackages."""
    root = source_root()
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*.py"))
    }


def snapshot_sources(destination, bindings):
    """Copy the verified producing package without flattening import directories.

    The caller records bindings before training starts. Rechecking each copied file
    prevents a concurrent source edit from producing a mixed-version snapshot.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for relative, expected in bindings.items():
        source = source_path(relative)
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Snapshot destination escapes its directory: " + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != expected:
            raise ValueError(
                "Refusing to overwrite different snapshot source: " + relative
            )
        shutil.copyfile(source, target)
        if file_sha256(target) != expected:
            raise ValueError("Source changed while snapshotting: " + relative)
