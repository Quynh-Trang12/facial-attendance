# Second live run: protocol

The first two live runs had one person at the webcam (An) and one photo-enrolled
template (Que, a different person), and no attempts by anyone unregistered. This run
measures the quality-gated check-in against the single-frame rule with several
webcam-enrolled people and an unregistered person, under the limits set from the
validation images (`calibration.json`).

## Setup

| Item | Value |
| --- | --- |
| Checkpoint | `../fac-05-inputs/deploy.pt`, sha256 `76f123382f8bf4812b0c3db4cf69e0c31aaf1d1f82bec193bc610bd7ba08145b` |
| Galleries, logs, trace | `artifacts/check-in-demo/live-2/` (new and empty; git-ignored) |
| Deciding crop | `box`, the bare detector box. The trace still scores every frame under all four crop rules. |
| Quality limits | `configs/check-in-quality.toml`, from `calibration.json` |
| Decision rule | Best score at least τ from the checkpoint. Each decision also logs the runner-up score, so a margin rule can be applied to the same attempts afterwards. |

Everyone at the camera agrees to take part. Only templates, scores and quality
numbers are saved, never frames.

## People

- At least three people enrolled **from the webcam**, An included. Photo enrollment is not used in this run.
- At least one person who is not enrolled, logged as `impostor`.

## Steps

Run from the worktree with the group's Python environment, in one shell that first
sets the checkpoint and galleries folder:

```bash
CKPT=../fac-05-inputs/deploy.pt G=artifacts/check-in-demo/live-2; export CKPT G
```

1. Enroll each person (5 webcam frames, SPACE to capture):

   ```bash
   ../py313-env/bin/python scripts/run_check_in_demo.py enroll --checkpoint "$CKPT" --galleries "$G" --name An
   ```

   Do not pass `--allow-similar`. If enrollment refuses someone as a duplicate of an
   enrolled person, write that down: it is a result.

2. For each enrolled person, 15 attempts in normal conditions, then 5 hard ones
   (head turned about 30°, light behind the head or dimmed, slow movement):

   ```bash
   ../py313-env/bin/python scripts/run_check_in_demo.py run --checkpoint "$CKPT" --galleries "$G" --mode both --crop box --person An
   ```

3. The unregistered person, 15 attempts:

   ```bash
   ../py313-env/bin/python scripts/run_check_in_demo.py run --checkpoint "$CKPT" --galleries "$G" --mode both --crop box --person impostor
   ```

4. Results into this folder:

   ```bash
   ../py313-env/bin/python scripts/run_check_in_demo.py summarize --log "$G/attempts-paired.csv" --output evidence/quality-gated-check-in/live-2-summary.json
   ```

   ```bash
   ../py313-env/bin/python scripts/run_check_in_demo.py diagnose --galleries "$G" --output evidence/quality-gated-check-in/live-2-diagnosis.json
   ```

   ```bash
   ../py313-env/bin/python scripts/run_check_in_demo.py replay --trace "$G/trace.csv" --output evidence/quality-gated-check-in/live-2-replay.json
   ```

## What the run answers

- Single against fused on the same frames: genuine accepts, misidentifications,
  impostor accepts, attempts with no decision, decision time.
- How many frames the gate refuses under these limits, and whether the refused
  frames score lower.
- With the runner-up score: the same decisions under τ plus a margin.
