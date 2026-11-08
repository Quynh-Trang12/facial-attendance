# UI-01 Gallery Base — Design and Implementation Notes

These are the historical UI-01.2 baseline notes. See the separate
[local UI-01.4 revisions](ui-01-4-gallery-revisions.md) for the current v0.2 API,
compatibility decisions and regression results; the original evidence below is retained.

Status: Draft implementation; review pending for UI-01.2 (Quế An) and UI-01.4 (chị Dung).

## Scope
Model-independent gallery storage only: identity IDs/names, multiple externally supplied embeddings, JSON persistence and metadata compatibility checks.
The storage component does not perform webcam capture, face detection, image quality checks, liveness checks, embedding inference, similarity matching, or attendance verification.
Model dimensions and IDs are supplied by the caller and must be confirmed against UI-01.1/SYS-02 before integration.

## Implementation
- src/facial_attendance/ui/gallery.py: FaceGallery, immutable Identity records, register/add_embedding/get, validated JSON save/load.
- tests/test_gallery.py: 11 source-only unit tests using synthetic vectors.
- Proposed ignored local storage destination: artifacts/gallery/gallery.json (caller-provided path; no shared configuration changed).
- Duplicate person ID creation is rejected; duplicate names are allowed. Updating an existing person's vectors is an explicit operation.
- Failed save or incompatible/corrupt gallery does not silently overwrite the original data.

## Local verification
From repository root on Windows PowerShell:
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_gallery.py' -v

On Linux/macOS:
PYTHONPATH=src python -m unittest discover -s tests -p 'test_gallery.py' -v

Isolated gallery tests: 11/11 passed in a source-only local workspace (not the complete repository environment).
Full existing test suite, UI, camera, recognition, and liveness integration have not been verified.

## UI-01.4 review preparation — not corrections evidence
Review the duplicate-ID policy, storage preservation, model metadata, and failure handling against the reviewed design.
Actual UI-01.2 findings: pending Quế An.
Actual UI-01.4 findings and confirmation: pending chị Dung.
UI-01.4 corrections: none attributed to review at this stage. If review finds no issue, record 'No change needed' with genuine review evidence.

## Pending decisions and limitations
Embedding dimension, normalization, model identity/version policy, distance metric, match threshold, recognition schema, and liveness gating are not finalized.
No file locking, gallery migration, duplicate-face matching, or model-specific inference is implemented. These are future design/integration decisions, not invented acceptance requirements.

Related Notion: https://app.notion.com/p/UI-01-2-Gallery-Schema-Registration-Flow-Draft-v0-1-40f863094ebe80f9b748c0fb79fe9262?source=copy_link
