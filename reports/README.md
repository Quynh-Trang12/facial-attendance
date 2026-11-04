# Verification evidence

Start with these reports for the current refactored code:

| Report | What was actually exercised |
| --- | --- |
| [Default-config data run](data/config-default-audit-2026-11-04.json) | Assigned dataset, with no path override, invoked from another working directory |
| [Fresh CPU-wheel run](environment/cpu-wheel-clean-install-2026-11-04.json) | Separate install using actual +cpu wheels, no CUDA build; finite CPU training/inference |
| [Colab CPU with real images](environment/colab-cpu-real-2026-11-04.json) | 56 original competition images; loader, synthetic environment step and real-image optimizer/inference; [39-test log](environment/colab-cpu-real-tests-2026-11-04.log) |
| [Colab Tesla T4 with real images](environment/colab-cuda-real-2026-11-04.json) | Same input subset; CUDA 12.6, real-image model/batch on `cuda:0`, finite backward/update/inference; [39-test log](environment/colab-cuda-real-tests-2026-11-04.log) |
| Documented Colab cells: [CPU](environment/colab-cpu-guide-review-2026-11-04.json), [CUDA](environment/colab-cuda-guide-review-2026-11-04.json) | Five setup-guide cells executed in IPython, using the documented local-data override instead of remote Kaggle authentication |
| [Remote Colab CPU run](environment/colab-cpu-retest-2026-11-04.json) | Python 3.13.18, CPU-only wheels, finite optimizer/inference and generated-image loader checks; [39-test log](environment/colab-cpu-tests-2026-11-04.log) |
| [CPU/IPython run](environment/cpu-notebook-audit-2026-11-04.json) | Explicit CPU optimizer step and inference using captured notebook-style subprocess output |
| [CUDA owner run](environment/cuda-strict-audit-2026-11-04.json) | Actual host GPU execution, exact owner patch and all 58 snapshot package versions |

Each report includes the source/configuration hashes used by its execution. Earlier runs remain useful history, but their source hashes can differ after a rename or refactor. Read their date/version/input source rather than assuming that every PASS belongs to the current revision. Fixture reports use generated images; they are not competition-data results.

The current unit suite has 39 passing tests locally and on CPU/GPU Colab. Run it using the README command to obtain your own results. Full-dataset acquisition/layout inspection was executed locally; current Colab runs used a real-image subset, while earlier Colab reports explicitly used generated fixtures. The first Colab environment check failed on an inherited notebook Matplotlib backend; its [failure report](environment/colab-cpu-first-failure-2026-11-04.json) is retained. Successful repeats explicitly selected the noninteractive Agg backend. Remote GitHub CI and human reviewer acceptance remain unverified.

The [real-image input manifest](data/colab-real-input-2026-11-04.json) identifies the source archive, all 56 original-image hashes and the 16 selected official trials. CPU/GPU acceptance reports contain the same archive hash and report 56 verified image hashes. The loader results validate all selected trial paths/labels and decode first batches, not every image in the competition.

The setup/data-loading review checked step order, interpreter selection, explicit device choice, token handoff to subprocesses, config/override consistency, stage logs and failure propagation. The revised guide's five Python cells were executed on both remote runtimes; only the explicitly documented device and existing-data selections differed. The guides separate acquisition, tests and model-runtime checks and explain why train-only class mapping, official trial order and seeded pair sampling are preserved. No notebook deliverable was created, so the skill's notebook-specific presentation checker was not applicable. Readability was reviewed manually; successful execution does not guarantee every platform or future Colab image is error-free. The Colab Secrets/API branch was not run remotely, and no owner token was uploaded. Both temporary Colab runtimes were terminated after evidence retrieval.
