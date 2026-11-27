# Check-in test guide: margin rule and quality-gated check-in

This guide tells the team how to test the new face decision rule, how to switch the
innovative feature off, and how to bring what works into the webcam app.

Two changes are involved, and they can be adopted separately:

| Change | What it does | Where it lives |
| --- | --- | --- |
| **Margin rule** | A face is named only if its best score reaches τ (0.462, from the checkpoint) **and** beats every other enrolled person by 0.15. Otherwise the result is unknown. This targets the app's known fault: an unregistered person, or a look-alike, being named as someone enrolled. | `src/facial_attendance/recognition/decision.py`, `configs/recognition.toml` (PR #21) |
| **Quality-gated check-in** (innovative feature) | Each frame must pass a quality gate (size, sharpness, brightness, contrast, head pose). Up to 3 good frames within 2 s are fused into one probe, and frames that disagree are rejected. The user sees a prompt such as "too dark" or "turn to the camera". | `quality.py`, `probe_fusion.py`, `check_in_live.py`, `configs/check-in-quality.toml` |

The webcam app on `feature/ui03-ui04-emotion-integration` uses neither yet. Its
`GalleryMatcher.match` names the best match whenever the score reaches τ. Until step 3
is done, test both changes with the standalone runner `scripts/run_check_in_demo.py`
and not with the app.

## 1. Switch the innovative feature off

The runner chooses the rule per run, so nothing needs editing:

| You want | Run with |
| --- | --- |
| The original model as the app uses it: one frame, best score ≥ τ | `--mode single --decision configs/recognition-tau-only.toml` |
| The original model plus the margin rule | `--mode single` (the margin in `configs/recognition.toml` is the default) |
| The innovative feature: gate, fusion and margin | `--mode fused` |
| Both on the same frames, for a fair comparison | `--mode both` |

`--mode single` skips the quality gate and the fusion entirely. It embeds the first
frame with exactly one face, as the app does today.

When the feature is merged into the app (step 3), the switch is `enabled` under
`[check_in]` in `configs/check-in-quality.toml`. It is `false` by default, so the app keeps
its single-frame rule unless someone turns it on.

## 2. Test the new model rule

### Setup (once)

1. Use the group's Python 3.13 environment from [setup](setup.md), opened at the
   repository root on branch `innovation/quality-gated-check-in`.
2. Get the face checkpoint from the team: FAC-05's `deploy.pt`. It is the same file as the
   app's `artifacts/models/face-recognition-resnet18.pt` (metric ResNet-18, model version
   `cb38ea47ceff`, τ 0.4619967). The runner refuses a checkpoint without a threshold.
3. Pick a new, empty gallery folder. The runner creates it, and git ignores everything
   under `artifacts/`.

```bash
CKPT=artifacts/models/face-recognition-resnet18.pt G=artifacts/check-in-demo/team-test; export CKPT G
```

Add `--download-detector` to the first command if the YuNet file is missing.

### Enroll

Each enrolled person runs this once. The window opens the webcam. Press SPACE five
times, once for each capture, and press Q or Esc to quit.

```bash
python scripts/run_check_in_demo.py enroll --checkpoint "$CKPT" --galleries "$G" --name <name>
```

Enroll at least two people **from the webcam**. Enrollment refuses a person whose
template is too close to someone already enrolled. Write that refusal down, because it
is a result.

### Check in

Press SPACE to start each attempt. `--person` records who is really in front of the
camera, so the summary can tell right from wrong.

```bash
python scripts/run_check_in_demo.py run --checkpoint "$CKPT" --galleries "$G" --mode both --person <name>
```

Someone who is **not** enrolled runs the same command with `--person impostor`. This
is the case the app gets wrong today, so it matters most.

The suggested amount is 15 normal attempts for each enrolled person and 5 hard ones
(head turned, light behind, moving). The impostor makes 15 attempts. The full protocol
is in [live-run-2-protocol.md](../evidence/quality-gated-check-in/live-run-2-protocol.md).

To check the original rule on its own, repeat a few attempts with
`--mode single --decision configs/recognition-tau-only.toml`.

### Read the result

```bash
python scripts/run_check_in_demo.py summarize --log "$G/attempts-paired.csv" --output "$G/summary.json"
```

The summary has one block per mode (`single`, `fused`):

| Field | Good result |
| --- | --- |
| `impostor_accept_rate` | 0: the unregistered person is never named as someone enrolled |
| `misidentified` | 0: no enrolled person is named as someone else |
| `genuine_accept_rate` | High. An unknown now and then is acceptable, because the margin trades some of these for safety. |
| `timeouts` (fused) | Few. Many timeouts mean the quality limits are too strict for the room. |
| `decision_ms_median` | How long a check-in takes |

Send the summary JSON to An. It holds names and counts, but no images or templates.
Keep the gallery folder local, because it holds biometric templates.

## 3. Bring it into the webcam app

The app belongs to Vỹ, so Vỹ makes or reviews these changes. Do them in this order and
stop after any step that is enough.

### 3a. Margin rule (fixes the wrong-name fault)

1. Merge PR #21 into `main`, then merge `main` into `feature/ui03-ui04-emotion-integration`.
   This adds `decision.py`, `configs/recognition.toml` and `tests/test_decision.py`, which
   are new files with no conflicts.
2. In `src/facial_attendance/recognition/gallery_matching.py`, have `GalleryMatcher.match`
   score every person and let `decide()` choose the result:

   ```python
   from .decision import decide, load_margin

   # in __init__: self.margin = load_margin("configs/recognition.toml")
   scores = [cosine(embedding, p.embeddings[0]) for p in self._gallery.people]
   d = decide([p.person_id for p in self._gallery.people], scores,
              self.contract.cosine_threshold, self.margin)
   person = next((p for p in self._gallery.people if p.person_id == d.label), None)
   return Match(d.label, person.name if person else None, d.best_score, d.accepted)
   ```

   Resolve the config path from the repository root, as `configs/project.toml` is.
3. **Keep the duplicate check in registration on τ alone.** `ControlledTestEnrollment.commit`,
   which the app's registration calls, uses `self.match(template)` to refuse a duplicate.
   With the margin, a new person who looks like two enrolled people would come back unknown
   and be saved as a new profile. Give the duplicate check its own plain best-score ≥ τ test.
4. Add tests in `tests/` like `test_decision.py`. Cover a clear winner that is accepted,
   a close runner-up that returns unknown, and a single enrolled person who needs ≥ τ + 0.15.
5. Re-run the "Check rejection" walkthrough in `docs/attendance-demo.md` with an
   unregistered person.

### 3b. Crop rule (only if the team's test supports it)

The app crops a square 1.2 × the detector box. In the crop study
([crop-study-diagnosis.json](../evidence/quality-gated-check-in/crop-study-diagnosis.json)),
square 1.2 made two people's templates close (cosine 0.88), and one of them was named as
the other in 19 % of frames. The bare box cut that to 0.51 and 0 %. The study had only two
people, so change the crop only if the team's runs confirm it. A crop change is a
preprocessing change: bump `PREPROCESSING_VERSION` in `recognition/inference.py` and
enroll everyone again.

### 3c. Quality-gated check-in (the innovative feature)

Do this only if step 2 shows fused does at least as well as single plus the margin.

1. Merge `innovation/quality-gated-check-in` (or cherry-pick `quality.py`,
   `probe_fusion.py` and `configs/check-in-quality.toml`).
2. The attendance coordinator already collects five consecutive live samples over at
   least one second for the liveness gate. For each of those frames, call
   `quality.measure(crop, landmarks)` and `quality.check(...)`, then pass the embedding
   (or `None` for a rejected frame) to `ProbeCollector.add(embedding, reasons, now)`.
3. On `Ready`, decide on `Ready.probe` with `decide()` from step 3a. On `Retry`, show
   `quality.prompt(...)` and start a new attempt. On `Collecting`, keep waiting.
4. Read `enabled` from `[check_in]` and keep today's single-capture path when it is
   `false`. This is the switch from section 1.
5. The liveness rule stays as it is: the fused identity must come from frames of the same
   gate-passing attempt.

## Known limits

- The margin of 0.15 comes from three people in one session. Change it in
  `configs/recognition.toml` only after the team's runs; the log keeps each runner-up
  score, so old attempts can be decided again under another margin.
- The quality limits come from the validation images. In a dark or bright room they may
  refuse most frames. That shows up as many fused timeouts.
- None of this has been tested inside the app yet.
