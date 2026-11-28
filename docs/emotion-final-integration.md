# Final emotion integration

Use [the canonical final record](emotion-finalized.md). Host input: one detected human-face crop. The adapter accepts uint8 HxWx3 RGB via predict_rgb or OpenCV BGR via predict_bgr; it handles all preprocessing internally. No frame, normalized tensor or alphabetical labels. Output: status, class_id, label, confidence, probabilities (Sad/Happy/Neutral). No calibrated confidence threshold is defined.

```python
from facial_attendance.emotion.final import EmotionPredictor

emotion = EmotionPredictor(checkpoint="artifacts/models/emotion/best.pt")
result = emotion.predict_bgr(face_crop_bgr)  # load once; call off the UI thread
```

Run from the installed project or set PYTHONPATH=src. [The controller example](../src/facial_attendance/emotion/final/example_ui_integration.py) samples once per three seconds, runs inference in a worker, and discards outputs for an earlier face generation. Call submit/poll on the UI thread. Pass None immediately when no valid single face exists; clear stale labels/scores. Supply the current track ID where available. Clear results on errors and person changes. Waiting within the interval may retain the latest sampled result for the same face only. Call close on shutdown.

Host responsibilities: detection, crop quality, identity, liveness, attendance gating. The final classifier does not detect multiple faces or assess pose/occlusion by itself. Emotion is informational, never attendance eligibility. The standalone local tester used approximate YuNet geometry checks; that is not evidence of a validated shared quality gate.

Checkpoint check (synthetic inference, not accuracy):
```text
python scripts/check_emotion_final.py --checkpoint artifacts/models/emotion/best.pt
```

Owner checks reproduced saved probabilities for each class, RGB/BGR agreement, three-second scheduling, invalid-input rejection and stale-result invalidation. [Verification](../reports/emotion/final/verification.json). An informal single-user webcam trial was reported. Vy must still independently load the checkpoint and test the integrated application; none of the pending UI-04.7/SYS-07 cases are marked passed here.
