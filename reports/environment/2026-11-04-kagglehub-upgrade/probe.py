"""One-step post-upgrade project and standalone Keras compatibility check."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import traceback

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
from facial_attendance.environment import check_environment, source_hashes

report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL",
          "python": platform.python_version(), "source_sha256": source_hashes(root),
          "probe_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
code = 1
try:
    report["project"] = check_environment(root)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf
    import keras

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Keras check requires GPU access; CPU fallback is disabled")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    keras.utils.set_random_seed(42)
    with tf.device("/GPU:0"):
        model = keras.Sequential([
            keras.Input((16, 16, 3)),
            keras.layers.Conv2D(4, 3, activation="relu", kernel_initializer=keras.initializers.Constant(.01)),
            keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(2),
        ])
        optimizer = keras.optimizers.Adam(learning_rate=.001)
        before = [weight.numpy().copy() for weight in model.trainable_weights]
        x = tf.ones((4, 16, 16, 3))
        y = tf.constant([0, 1, 0, 1])
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
        if not all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
            raise RuntimeError("Keras produced missing or nonfinite gradients")
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        changed = sum(bool((old != new.numpy()).any()) for old, new in zip(before, model.trainable_weights))
        output = model(x, training=False)
    if (not changed or not bool(tf.math.is_finite(loss)) or list(output.shape) != [4, 2]
            or "GPU:0" not in output.device or not bool(tf.reduce_all(tf.math.is_finite(output)))):
        raise RuntimeError("Keras training/inference assertions failed")
    report["keras"] = {"version": keras.__version__, "tensorflow": tf.__version__,
                       "backend": keras.backend.backend(), "loss": float(loss),
                       "gradients_finite": True, "updated_parameter_tensors": changed,
                       "inference_shape": list(output.shape), "inference_finite": True,
                       "device": output.device}
    report["status"] = "PASS"
    code = 0
except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
    report["traceback"] = traceback.format_exc()

out = root / "reports/environment/2026-11-04-kagglehub-upgrade/result.json"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")
print(json.dumps(report, indent=2))
raise SystemExit(code)
