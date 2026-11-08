# Architecture

Keep a single Python application with one package, `facial_attendance`. Training and inference code belong beside the relevant feature, so the same preprocessing and checkpoint interpretation can be reused by evaluation and the UI. Attendance is the first application; the repository name does not restrict future uses of the recognition, liveness and emotion modules.

| Area | Responsibility | Next implementation |
| --- | --- | --- |
| `recognition/` | Classification/pair datasets, Kaggle acquisition and a training-pair sampler; future embeddings/evaluation | Review split/leakage protocol, then implement two baselines |
| `liveness/` | Live/spoof data, trained anti-spoofing model and inference | Review [LIV-01 protocol](liveness-protocol.md), acquire/audit selected data, then implement training |
| `emotion/` | Public expression data, trained classifier and label mapping | Dataset audit and per-class evaluation |
| `ui/` | Webcam lifecycle, registration, identity feedback and display of all modules | UI prototype and agreed module adapters |
| `config.py` | Resolve shared data/output locations from TOML | Add settings only when their consuming feature exists |
| `environment.py` | Version/device policy, CLI and JSON evidence; no eager ML imports | Independent developer reproduction on an explicit CPU/CUDA target |
| `runtime_checks.py` | ML imports, optimizer/inference and image/metric API smoke checks | Loaded once when the environment check executes |

The environment CLI loads `runtime_checks.py` at one documented boundary so help, policy tests and failed-install reports work without an installed ML stack. Computational helpers keep their imports at module level.

Recognition now contains working `data_loader.py` and `download.py` implementations described in [data loading](data-loading.md). Liveness, emotion and UI remain package boundaries whose `__init__.py` files do not implement the features. Add `data_loader.py`, `model.py`, `train.py` or `evaluate.py` inside a feature only as it gains working code. Put reusable tests under `tests/`; keep command entry points in `scripts/` and implementation in `src/`.

The intended runtime captures a webcam face, obtains the agreed face crop, evaluates liveness, verifies the identity and predicts emotion, then updates UI feedback. Registration adds a new identity and its embeddings through the recognition module. The exact crop, channel order, input dimensions, normalization, output schema and failure behavior must be agreed under SYS-02; this scaffold does not silently settle those contracts. Liveness must be integrated with attendance decisions rather than displayed as an unrelated classifier.

The selected liveness baseline is one trained/fine-tuned RGB CNN operating on a face image. Its prediction score indicates spoof likelihood, separate from identity similarity. The application blocks attendance for spoof and shows a warning; a valid live result and verified identity are required to record attendance. No motion-analysis or frame-aggregation component is required by this design. The [protocol](liveness-protocol.md) defines the data and webcam checks; implementation remains pending.

`configs/project.toml` resolves relative paths from the repository root and also permits explicit external dataset locations. Its default face_data now points at the actual Kaggle cache; the CLI obtains local/API defaults from this one config. Reading it does not create directories or assert that datasets exist. Generated checkpoints and attendance records go under ignored `artifacts/`; small run results and plots go under `reports/`. Keep machine paths and measured hardware in execution evidence or setup instructions, not feature code.

There is no web service, database layer, plugin registry or deployment stack yet. Add one only if a concrete application requirement needs it. Experimental notebooks may be introduced later for analysis, while application behavior remains in the Python package.

For future workplace sentiment analysis or mental-health monitoring, build temporal aggregation and reporting as a new application feature that consumes emotion predictions; keep it separate from both the emotion model and attendance records. Those extensions are not implemented or required to close SYS-03, so there are no empty analytics services or speculative storage schemas in this scaffold.
