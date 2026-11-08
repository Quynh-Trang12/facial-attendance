"""Whole-dataset label, duplicate and leakage audit for the face competition data."""

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

from .data_loader import (
    IMAGE_SUFFIXES,
    SPLITS,
    ClassificationFaces,
    FaceLayout,
    read_verification_pairs,
    sample_training_pairs,
)

VERIFICATION = "verification_data"
# All four partitions are independent, so no two of them may share image bytes.
LEAKAGE_PAIRS = (
    ("train_data", "val_data"),
    ("train_data", "test_data"),
    ("val_data", "test_data"),
    ("train_data", VERIFICATION),
    ("val_data", VERIFICATION),
    ("test_data", VERIFICATION),
)
# Who keeps an image found in several partitions. Verification stays whole because the
# required AUC uses every official trial; val/test hold only two images per identity,
# while train can spare a copy.
KEEP_ORDER = (VERIFICATION, "test_data", "val_data", "train_data")
SAMPLED_PAIRS_PER_LABEL = 50_000
EXAMPLE_LIMIT = 20
HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class ImageRecord:
    partition: str
    identity: str | None  # None for verification_data, which has no identity folders.
    path: Path
    sha256: str


def file_sha256(path: Path) -> str:
    """Hash in chunks so memory stays flat however large a file is."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(path: Path) -> tuple[tuple[int, int] | None, str | None, str | None]:
    """Return (size, mode, error); a decoder error is a finding, not a crash."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            # open() only reads the header; load() decodes every pixel.
            image.load()
            return image.size, image.mode, None
    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def _verification_images(layout: FaceLayout, trials) -> list[Path]:
    """List every verification image, including ones no trial uses."""
    base = (layout.root / VERIFICATION).resolve()
    # Trial files are always included, whatever their suffix; a set keeps a
    # symlink to another verification file from being counted as a duplicate.
    files = sorted(
        {
            p.resolve()
            for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
        | {p for t in trials for p in (t.left, t.right)}
    )
    # Same boundary rule as the loader: a symlink may not pull in outside images.
    for path in files:
        if not path.is_relative_to(base):
            raise ValueError(f"Image path escapes {VERIFICATION}: {path}")
    return files


def _identity_stats(dataset: ClassificationFaces) -> dict:
    counts = Counter(sample.identity for sample in dataset.samples)
    values = sorted(counts.values())
    return {
        "images": len(dataset),
        "identities": len(counts),
        "images_per_identity": {
            "min": values[0],
            "median": median(values),
            "max": values[-1],
            "singletons": sum(1 for v in values if v == 1),
        },
    }


class _Auditor:
    """Collect per-file facts once, then derive every check from those facts."""

    def __init__(self, layout: FaceLayout, progress: Callable | None):
        self.layout = layout
        self.progress = progress
        self.records: list[ImageRecord] = []
        self.decode_failures: list[dict] = []
        self.sizes = defaultdict(Counter)
        self.modes = defaultdict(Counter)

    def relative(self, path: Path) -> str:
        # Relative paths keep machine-specific home directories out of reports.
        return path.relative_to(self.layout.root).as_posix()

    def inspect(self, items: list[tuple[str, str | None, Path]]):
        for done, (partition, identity, path) in enumerate(items, start=1):
            self.records.append(
                ImageRecord(partition, identity, path, file_sha256(path))
            )
            size, mode, error = _decode(path)
            if error is not None:
                self.decode_failures.append(
                    {"path": self.relative(path), "error": error}
                )
            else:
                self.sizes[partition][f"{size[0]}x{size[1]}"] += 1
                self.modes[partition][mode] += 1
            if self.progress is not None:
                self.progress(done, len(items))

    def duplicates(self) -> tuple[dict, dict, dict]:
        by_digest = defaultdict(list)
        for record in self.records:
            by_digest[record.sha256].append(record)
        within = {"groups": 0, "redundant_files": 0, "examples": []}
        across_ids = {"groups": 0, "examples": []}
        across_parts = {
            f"{a}/{b}": {"shared_images": 0, "examples": []} for a, b in LEAKAGE_PAIRS
        }
        for digest, group in sorted(by_digest.items()):
            if len(group) < 2:
                continue
            paths = [self.relative(r.path) for r in group]
            example = {"sha256": digest, "paths": paths}
            per_identity = Counter(
                (r.partition, r.identity) for r in group if r.identity is not None
            )
            repeats = [count for count in per_identity.values() if count > 1]
            if repeats:
                within["groups"] += len(repeats)
                within["redundant_files"] += sum(repeats) - len(repeats)
                _keep(within["examples"], example)
            if len({r.identity for r in group if r.identity is not None}) > 1:
                across_ids["groups"] += 1
                _keep(across_ids["examples"], example)
            partitions = {r.partition for r in group}
            for a, b in LEAKAGE_PAIRS:
                if a in partitions and b in partitions:
                    entry = across_parts[f"{a}/{b}"]
                    entry["shared_images"] += 1
                    _keep(entry["examples"], example)
        return within, across_ids, across_parts


def _keep(examples: list, item):
    # Counts stay exact; only the listed examples are capped to bound report size.
    if len(examples) < EXAMPLE_LIMIT:
        examples.append(item)


def exclusion_entries(records: list[ImageRecord], relative: Callable) -> list:
    """Return sorted (path, reason) so one copy of each image survives, in one partition."""
    by_digest = defaultdict(list)
    for record in records:
        by_digest[record.sha256].append(record)
    entries = []
    for group in by_digest.values():
        owner = min((r.partition for r in group), key=KEEP_ORDER.index)
        kept = set()
        for record in sorted(group, key=lambda r: r.path):
            if record.partition == VERIFICATION:
                continue
            if record.partition != owner:
                reason = owner
            elif record.identity in kept:
                reason = "within_identity"
            else:
                kept.add(record.identity)
                continue
            entries.append((relative(record.path), reason))
    return sorted(entries)


def _write_exclusions(file: Path, entries: list, manifest: str):
    file.parent.mkdir(parents=True, exist_ok=True)
    # Never overwritten: a new list changes what every model is trained and scored on.
    with file.open("x", encoding="utf-8") as handle:
        handle.write(
            "# Classification images the loader must not use. Do not edit by hand; "
            "regenerate with scripts/audit_face_data.py --raw --write-exclusions.\n"
            f"# unfiltered dataset_manifest_sha256={manifest}\n"
            "# columns: path, partition that keeps the image or within_identity\n"
        )
        handle.writelines(f"{path}\t{reason}\n" for path, reason in entries)


def _sampler_check(train, records: list[ImageRecord], *, seed: int) -> dict:
    """Hash-check what the training sampler really draws, not only the pool it reads."""
    digests = {r.path: r.sha256 for r in records}
    held_out = {r.sha256 for r in records if r.partition != "train_data"}
    pairs = sample_training_pairs(
        train, count_per_label=SAMPLED_PAIRS_PER_LABEL, seed=seed
    )
    images = {path for pair in pairs for path in (pair.left, pair.right)}
    return {
        "seed": seed,
        "pairs": len(pairs),
        "distinct_images": len(images),
        "images_shared_with_held_out": sum(digests[p] in held_out for p in images),
        "excluded_images_drawn": sum(p in train.layout.excluded for p in images),
        "identical_positive_pairs": sum(
            pair.label == 1 and digests[pair.left] == digests[pair.right]
            for pair in pairs
        ),
    }


def _known_identities(records: list[ImageRecord], trials) -> dict:
    """Count verification images and trials whose person a byte-identical copy reveals.

    Verification images have no identity labels, so this is a lower bound on how often
    the official trials involve people the model trains on. It needs the unfiltered data:
    filtering removes the classification copies that name the person.
    """
    identities_by_digest = defaultdict(set)
    for record in records:
        if record.identity is not None:
            identities_by_digest[record.sha256].add(record.identity)
    named = {
        record.path: identities_by_digest[record.sha256]
        for record in records
        if record.partition == VERIFICATION and record.sha256 in identities_by_digest
    }
    scored = {path for trial in trials for path in (trial.left, trial.right)} & set(named)
    sides_named = [(trial.left in named) + (trial.right in named) for trial in trials]
    return {
        "is_lower_bound": True,
        "verification_images_named": len(named),
        "identities_named": len(set().union(*named.values())),
        "scored_images_named": len(scored),
        "identities_named_by_scored_images": len(
            set().union(*(named[path] for path in scored))
        ),
        "trials": len(trials),
        "trials_with_a_named_image": sum(count > 0 for count in sides_named),
        "trials_with_both_images_named": sum(count == 2 for count in sides_named),
    }


def _trial_checks(trials, digests: dict[Path, str], relative: Callable) -> dict:
    labels = Counter(t.label for t in trials)
    keys = defaultdict(list)
    self_pairs, identical = [], Counter()
    # "trial" numbers count parsed trials from 1, skipping blank and comment lines.
    for number, trial in enumerate(trials, start=1):
        # Trials are unordered: (a, b) and (b, a) describe the same comparison.
        keys[tuple(sorted((trial.left, trial.right)))].append((number, trial.label))
        if trial.left == trial.right:
            self_pairs.append({"trial": number, "path": relative(trial.left)})
        elif digests[trial.left] == digests[trial.right]:
            identical[trial.label] += 1
    repeated = {"pairs": 0, "extra_trials": 0, "examples": []}
    conflicting = {"pairs": 0, "examples": []}
    for (left, right), entries in keys.items():
        if len(entries) < 2:
            continue
        example = {
            "left": relative(left),
            "right": relative(right),
            "trials": [{"trial": n, "label": label} for n, label in entries],
        }
        if len({label for _, label in entries}) > 1:
            conflicting["pairs"] += 1
            _keep(conflicting["examples"], example)
        else:
            repeated["pairs"] += 1
            repeated["extra_trials"] += len(entries) - 1
            _keep(repeated["examples"], example)
    used = {p for t in trials for p in (t.left, t.right)}
    return {
        "trials": len(trials),
        "label_counts": {str(k): v for k, v in sorted(labels.items())},
        "same_identity_fraction": round(labels[1] / len(trials), 4),
        "self_pairs": {
            "count": len(self_pairs),
            "examples": self_pairs[:EXAMPLE_LIMIT],
        },
        "identical_content_pairs": {
            "count": sum(identical.values()),
            "by_label": {str(k): v for k, v in sorted(identical.items())},
        },
        "repeated_pairs": repeated,
        "conflicting_labels": conflicting,
        "distinct_images_used": len(used),
        "verification_images_unused": len(digests) - len(used),
    }


def _spot_check(
    train: ClassificationFaces, trials, *, seed: int, size: int, relative
) -> dict:
    """A seeded sample a person can open and confirm by eye."""
    rng = random.Random(seed)
    samples = rng.sample(train.samples, min(size, len(train.samples)))
    picked = rng.sample(range(len(trials)), min(size, len(trials)))
    return {
        "seed": seed,
        "train_samples": [
            {"path": relative(s.path), "identity": s.identity, "target": s.target}
            for s in samples
        ],
        "trials": [
            {
                "trial": i + 1,
                "left": relative(trials[i].left),
                "right": relative(trials[i].right),
                "label": trials[i].label,
            }
            for i in sorted(picked)
        ],
    }


def audit_dataset(
    layout: FaceLayout,
    *,
    seed: int = 42,
    sample_size: int = 20,
    progress: Callable[[int, int], None] | None = None,
    write_exclusions: Path | None = None,
) -> dict:
    """Hash and decode every file; derive identity, duplicate, leakage and trial checks.

    Only paths, digests and counters are kept, so the full 397k-image dataset
    fits in a Colab runtime's memory.
    """
    train = ClassificationFaces(layout, "train_data")
    # The loader rejects any val/test identity missing from train; reaching the
    # next line therefore proves the subset rule for both held-out splits.
    splits = {"train_data": train}
    for split in SPLITS[1:]:
        splits[split] = ClassificationFaces(
            layout, split, class_to_idx=train.class_to_idx
        )
    trials = read_verification_pairs(layout)

    items = [
        (split, sample.identity, sample.path)
        for split, dataset in splits.items()
        for sample in dataset.samples
    ]
    items += [(VERIFICATION, None, p) for p in _verification_images(layout, trials)]
    auditor = _Auditor(layout, progress)
    auditor.inspect(items)
    within, across_ids, across_parts = auditor.duplicates()
    digests = {r.path: r.sha256 for r in auditor.records if r.partition == VERIFICATION}
    trial_report = _trial_checks(trials, digests, auditor.relative)

    sampler = _sampler_check(train, auditor.records, seed=seed)
    excluded_loaded = sum(d.excluded_count for d in splits.values())

    train_ids = set(train.class_to_idx)
    partitions = {split: _identity_stats(dataset) for split, dataset in splits.items()}
    for split in SPLITS[1:]:
        present = {s.identity for s in splits[split].samples}
        partitions[split]["subset_of_train"] = True
        partitions[split]["train_identities_absent"] = len(train_ids - present)
    partitions[VERIFICATION] = {"images": len(digests)}

    leaked = {
        k: v["shared_images"] for k, v in across_parts.items() if v["shared_images"]
    }
    failures = []
    if auditor.decode_failures:
        failures.append(f"{len(auditor.decode_failures)} image files cannot be decoded")
    if trial_report["conflicting_labels"]["pairs"]:
        failures.append(
            f"{trial_report['conflicting_labels']['pairs']} trial pairs appear with both labels"
        )
    for name, count in leaked.items():
        failures.append(f"{count} identical images are shared by {name}")
    if sampler["images_shared_with_held_out"] or sampler["excluded_images_drawn"]:
        failures.append(
            f"{sampler['images_shared_with_held_out']} sampled training images are also "
            f"held out and {sampler['excluded_images_drawn']} are excluded images"
        )
    if layout.excluded:
        # A stale list could drop nothing and still leave the rest of the data clean.
        if excluded_loaded != len(layout.excluded):
            failures.append(
                f"The loader dropped {excluded_loaded} of "
                f"{len(layout.excluded)} listed images"
            )
        if within["groups"]:
            failures.append(
                f"{within['redundant_files']} duplicate copies inside one identity "
                "remain after filtering"
            )
    warnings = []
    if within["groups"]:
        warnings.append(
            f"{within['redundant_files']} duplicate copies inside one identity; "
            "sampled positives can then compare identical images"
        )
    if across_ids["groups"]:
        warnings.append(
            f"{across_ids['groups']} images appear under more than one identity "
            "(possible label errors)"
        )
    if trial_report["self_pairs"]["count"]:
        warnings.append(
            f"{trial_report['self_pairs']['count']} trials compare a file with itself"
        )
    if trial_report["identical_content_pairs"]["count"]:
        warnings.append(
            f"{trial_report['identical_content_pairs']['count']} trials compare two "
            "byte-identical files"
        )
    if trial_report["repeated_pairs"]["pairs"]:
        warnings.append(
            f"{trial_report['repeated_pairs']['extra_trials']} trials repeat an earlier pair"
        )

    # One digest over every (path, content) line fingerprints the exact data audited,
    # so a Colab run can be compared with a local one without shipping images.
    manifest = sha256()
    for record in sorted(auditor.records, key=lambda r: r.path):
        manifest.update(f"{auditor.relative(record.path)}\t{record.sha256}\n".encode())
    report_exclusions = {}
    if write_exclusions is not None:
        entries = exclusion_entries(auditor.records, auditor.relative)
        _write_exclusions(write_exclusions, entries, manifest.hexdigest())
        report_exclusions = {
            "written_exclusions": {
                "entries": len(entries),
                "by_reason": dict(Counter(reason for _, reason in entries)),
            }
        }
    return report_exclusions | {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "files_audited": len(auditor.records),
        "dataset_manifest_sha256": manifest.hexdigest(),
        "identity_mapping_sha256": sha256(
            json.dumps(train.class_to_idx, sort_keys=True).encode()
        ).hexdigest(),
        "pairs_file_sha256": file_sha256(layout.pairs_file),
        "filtering": {
            "mode": "filtered" if layout.excluded else "unfiltered",
            "images_listed": len(layout.excluded),
            "images_dropped_by_loader": {
                split: dataset.excluded_count for split, dataset in splits.items()
            },
        },
        "sampler": sampler,
        "known_identities": None
        if layout.excluded
        else _known_identities(auditor.records, trials),
        "partitions": partitions,
        "duplicates": {
            "within_identity": within,
            "across_identities": across_ids,
            "across_partitions": across_parts,
        },
        "trials": trial_report,
        "images": {
            "decode_failures": {
                "count": len(auditor.decode_failures),
                "examples": auditor.decode_failures[:EXAMPLE_LIMIT],
            },
            "sizes": {k: dict(v.most_common()) for k, v in auditor.sizes.items()},
            "modes": {k: dict(v.most_common()) for k, v in auditor.modes.items()},
        },
        "spot_check": _spot_check(
            train, trials, seed=seed, size=sample_size, relative=auditor.relative
        ),
        "rules": {
            "fail": [
                "undecodable image",
                "trial pair with conflicting labels",
                "identical image shared across "
                + ", ".join(f"{a}/{b}" for a, b in LEAKAGE_PAIRS),
                "sampled training image that is held out or excluded",
                "filtered data only: listed image not dropped, or duplicate inside one identity",
            ],
            "warn": [
                "duplicate inside one identity",
                "image under more than one identity",
                "self-pair, identical-content or repeated trial",
            ],
        },
    }
