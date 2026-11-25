# Course requirements and coverage

The governing source is the supplied [project brief](reference/project-brief.pdf). Face-recognition work uses the assigned [11-785 Fall 2020 Homework 2 Part 2 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). The [WBS and Acceptance Criteria](https://docs.google.com/spreadsheets/d/14iyTbZmOgO3WBDBv0jlLX9f2WhTDOMOeW1GbgnflECw/edit) define task ownership and acceptance evidence.

| Requirement | Current coverage |
| --- | --- |
| Train and compare classification and metric-learning face embeddings | [FAC-05](https://github.com/Quynh-Trang12/facial-attendance/blob/ba17d14d8acb349fc1ab728b4396561bd0878fce/evidence/FAC-05/README.md) records both trained approaches and selects the metric model used by the GUI. |
| Evaluate ROC/AUC and similarity | FAC-05 records validation selection and comparison on the official pairs. Webcam-to-template accuracy is a separate check. The isolated Attendance experiment reproduced a wrong-person match. |
| Integrate anti-spoofing for live, printed and screen faces | AENet-Full epoch 18 runs in the webcam app. The recorded screen replay subset was rejected. Physical print and wider live/replay tests remain pending for this delivery. |
| Display trained facial expression predictions | The GUI displays the supplied Emotion ResNet-18 labels. Its outputs remain informational in the attendance flow. |
| Webcam recognition, unknown-person prompt, and registration | The local workspace supports unknown-person registration and five liveness-checked references. Physical macOS registration and a subsequent preview match were observed. |
| Store attendance | Manual and automatic local SQLite recording are available in the isolated test workspace, with one row per person per session. Production recording remains disabled. |
| Versioned setup and inference checks | [Runtime setup](setup.md) defines the supported Python and device profiles. See [verification evidence](../reports/README.md) for execution records. Native Windows webcam and independent reviewer checks are pending. |
| Report, continuous demonstration, and contribution submissions | The brief specifies a group report of at most 12 pages, a continuous 5–10 minute demonstration with balanced visible participation, and individual reports of at most five pages. Use the [demo guide](attendance-demo.md) as the walkthrough. |

The brief allows partial fine-tuning and disallows using a fully pretrained model without further training. Keep classification splits and verification trials in their assigned roles. [Data loading](data-loading.md) documents acquisition and verified loader coverage. The [demo verification](attendance-demo.md#verification) identifies the application checks that have been run and the limits that remain.
