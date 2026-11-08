# UI-01.4 — Local gallery revisions

Status: local implementation and verification for review by chị Dung. This is not task
completion or reviewer approval. No commit or pull request is created.

## Inspection and sources

The starting checkout was `main` at `c4e4812`; it did not contain the gallery.
Work uses the existing `feature/ui-01-gallery-base` at `7329057`, which adds
`gallery.py`, `test_gallery.py` and the UI-01.2 design notes. The existing
untracked `reports/environment/vy-independent-retest.json` is unrelated and
must remain unchanged. No other source callers of `FaceGallery` were found in
this branch; the gallery tests are its only executable callers.

Reviewed sources:

- [Original UI-01.2 notes](ui-01-gallery-base.md) and
  [UI-01.2 draft](https://app.notion.com/p/40f863094ebe80f9b748c0fb79fe9262).
- [UI-01.4 review preparation](https://app.notion.com/p/40f863094ebe80f98ed7df3074f48d11).
- [UI-01.1 embedding schema](https://github.com/Quynh-Trang12/facial-attendance/blob/docs/sys-04-experiment-settings/docs/embedding-schema.md).
- [SYS-02 v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d),
  especially sections 4.3, 8 and 9.1.

The UI-01.2/UI-01.4 pages still contain initial pending-review text. The
7 November PASS AS DRAFT result and three actual findings below were supplied
by the user; this work does not change those pages or claim new approval.

## Compatibility plan recorded before implementation

| Contract | Existing behavior | Required local correction |
| --- | --- | --- |
| Unique ID and name | Exact-case ID uniqueness, nonempty names | Casefold ID comparisons; preserve spelling; names 1–64 characters |
| Numeric embeddings | Lists/tuples of Python int/float only | Accept 1D numeric array-like vectors including NumPy float32; reject bool, text, nonfinite, nested or wrong-length inputs |
| Model metadata and 512D | Configurable dimension and exact model ID/version check already exist | Preserve these interfaces; integration passes checkpoint D=512, synthetic tests may use smaller D |
| Unit template | Arbitrary finite vectors and multiple samples | Validate norm within 1e-3 and components in [-1, 1]; exactly one supplied averaged template |
| Accepted enrollment count | Missing | Required integer `n_images` in [3, 10] |
| Registration timestamp | Any timezone-aware ISO timestamp | Store UTC, converting an explicit offset without changing the instant |
| Failed persistence | File preserved, pending memory retained | Roll back all pending registrations/template updates to last successful save or validated load |

These are storage-boundary changes. The caller still owns accepted-live sample
selection, `normalize(mean(normalize(e_i)))`, consistency/face-duplicate checks,
ID generation and liveness authorization. A count and normalized vector cannot
prove their provenance; the gallery must only be called after those checks.
No recognition, liveness inference, webcam or GUI work belongs in this change.

### Explicit format and API change

- Use `schema_version = "0.2"`; retain the five gallery header fields and
  `embeddings` list, now containing exactly one vector. Add only `n_images` to
  the identity record. This is incompatible with the v0.1 sample-list semantics.
- `register(person_id, name, embedding, *, n_images, created_at=None)` takes
  the already averaged normalized template and its accepted-image count.
- `update_template(person_id, embedding, *, n_images)` explicitly replaces the
  template and count, preserving the identity's original ID, name and timestamp.
- `add_embedding` cannot safely append a sample or reconstruct the average
  from an already normalized average. Retain a rejecting method with a clear
  instruction to use `update_template`; do not silently reinterpret it.
- `get`, registration, template updates and loading use `str.casefold()` for
  uniqueness/lookup. No accent stripping or Unicode normalization is added.
  Names are never uniqueness keys. Case-variant duplicate files fail as a whole.
- `register`/`update_template` stage changes. Only successful `save` means
  persistence succeeded. After a failed save, explicitly re-register/reapply
  the intended update before retrying. All changes since the last successful
  save/load are rolled back, including a first save reverting to an empty
  gallery. `from_dict` establishes a validated in-memory baseline, not proof
  that a file has been written. An instance is for one serialized writer and
  one logical gallery; cross-process locking is outside this change.

### Legacy data and decisions for the owner/reviewer

Do not load v0.1 as v0.2 or guess `n_images` from the old vector count: old
records do not establish that those vectors came from accepted live images or
that a single vector is an average. Loading v0.1 must raise an actionable format
error, without changing the file. Keep any existing v0.1 file intact. The owner
must choose verified manual migration from trustworthy enrollment evidence or
re-enrollment into a separate v0.2 file. No automatic migration is provided.
Changing recognition model weights requires re-enrollment as specified by
UI-01.1. Resolve casefold collisions explicitly; never silently merge people.

The user confirmed there are no known legacy data files or external callers
requiring compatibility and authorized this transition before implementation.
Chị Dung should review the API/format, casefold policy and rollback semantics.
Integration must supply the actual
checkpoint model ID, weights version and D=512, never the synthetic test values.
JSON serializes full Python float values; promoting float32 to Python float
preserves those values exactly and no short-decimal rounding is applied.

## Verification

Executed on 8 November 2026, Windows, Python 3.13.17, NumPy 2.2.6. PowerShell
commands from the repository root (the environment assignment was included
before every invocation):

```powershell
$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m unittest discover -s tests -p 'test_gallery.py' -v
$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m unittest discover -s tests -v
```

| Run | Command | Passed | Failed / errored tests | Skipped | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Original baseline, sandbox | Gallery command | 0 | 11 | 0 | 15 error reports, including cleanup errors: Windows denied temporary-directory operations |
| Original baseline, outside sandbox | Gallery command | 11 | 0 | 0 | Passed |
| Revised gallery, outside sandbox | Gallery command | 25 | 0 | 0 | Passed; all three NumPy tests executed |
| Revised complete suite, outside sandbox | Full-suite command | 71 | 0 | 0 | Passed |

The failed sandbox attempt is an environment failure, not a passing test run.
No dependencies were installed or changed. `git diff --check` also passed.

Regression coverage includes registration/update inputs (NumPy arrays and
scalars, dimensions, bool/text/complex, NaN/infinity), exact float32-value
round-trip with a 512D vector, casefold registration/lookup/update and duplicate
records on disk, first-save failure, rollback after load and `from_dict`,
serialization/mkdir/temp-file/fsync/replace failures, and explicit retry.
The reported failure sequence is exercised by saving EMP001, failing EMP002's
save, then successfully saving EMP003 and proving EMP002 is still absent from
both memory and the reloaded file. Existing file bytes are preserved on failure.
Atomic replacement remains the write strategy; temporary-file cleanup is best
effort if the operating system refuses deletion, so an orphan temporary file
may need manual removal after a storage failure.

The 11 original cases were retained (with template/count fixtures adjusted for
the documented contract), and 14 targeted cases were added. Unrelated tests and
historical academic evidence were not edited. The pre-existing report's SHA-256
remained `BC6D630E592FC7FEF894E153EE25C293929530999ED2325C2BA5D2BC69C28D23`.

## Remaining review and integration work

Chị Dung's independent review/reproduction and the UI-01.5 walkthrough remain
pending. This work does not certify averaging, image quality, liveness,
same-person consistency or recognition-based duplicate checks: those remain
caller responsibilities before `register` or `update_template`. Real model
metadata, D=512 and trained-model integration must be checked by the relevant
owners. The tests use synthetic embeddings and do not establish end-to-end
webcam or recognition behavior. No task status, external tracker or review
confirmation was changed.
