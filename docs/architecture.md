# Architecture

The repository is organized as one Python application with the package `facial_attendance`. Feature modules provide a shared location for training, inference, preprocessing and checkpoint interpretation across evaluation and the UI. Attendance is the first application; the repository name does not restrict future uses of the recognition, liveness and emotion modules.

| Area | Responsibility | Next implementation |
| --- | --- | --- |
| `recognition/` | Classification/pair datasets, Kaggle acquisition and a training-pair sampler; future embeddings/evaluation | Split/leakage protocol review and two baselines |
| `liveness/` | Live/spoof data, trained anti-spoofing model and inference | Source/split audit and print/replay protocol |
| `emotion/` | Public expression data, trained classifier and label mapping | Dataset audit and per-class evaluation |
| `ui/` | Webcam lifecycle, registration, identity feedback and display of all modules | UI prototype and agreed module adapters |
| `config.py` | Resolve shared data/output locations from TOML | Settings associated with implemented features |
| `environment.py` | Version/device policy, CLI and JSON evidence; no eager ML imports | Independent developer reproduction on an explicit CPU/CUDA target |
| `runtime_checks.py` | ML imports, optimizer/inference and image/metric API smoke checks | Loaded once when the environment check executes |

The environment CLI loads `runtime_checks.py` at one documented boundary so help, policy tests and failed-install reports work without an installed ML stack. Computational helpers keep their imports at module level.

Recognition now contains working `data_loader.py` and `download.py` implementations described in [data loading](data-loading.md). Liveness, emotion and UI remain package boundaries whose `__init__.py` files do not implement the features. Feature implementations can extend this layout with `data_loader.py`, `model.py`, `train.py` or `evaluate.py`. Reusable tests are located in `tests/`, command entry points in `scripts/` and implementation in `src/`.

The intended runtime captures a webcam face, obtains the agreed face crop, evaluates liveness, verifies the identity and predicts emotion, then updates UI feedback. Registration adds a new identity and its embeddings through the recognition module. SYS-02 tracks the crop, channel order, input dimensions, normalization, output schema and failure behavior. These interface contracts are separate from the environment scaffold. In the intended attendance flow, liveness informs the attendance decision.

`configs/project.toml` resolves relative paths from the repository root and also permits explicit external dataset locations. Its default face_data now points at the actual Kaggle cache; the CLI obtains local/API defaults from this one config. Reading it does not create directories or assert that datasets exist. Generated checkpoints and attendance records go under ignored `artifacts/`; small run results and plots go under `evidence/`. Setup documentation and execution evidence describe machine-specific paths and measured hardware.

The current implementation has no web service, database layer, plugin registry or deployment stack. Experimental notebooks may be introduced later for analysis, while application behavior remains in the Python package.

Potential workplace sentiment analysis or mental-health monitoring extensions would consume emotion predictions through a separate application feature for temporal aggregation and reporting. These extensions are not implemented and are outside SYS-03.
