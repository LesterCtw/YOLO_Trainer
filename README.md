# YOLO Trainer

## Current status

This repo has the initial project scaffold and the first YOLO Training Project
workflow for the YOLO Trainer MVS.

- GitHub repository: <https://github.com/LesterCtw/YOLO_Trainer>
- Agent configuration lives in `AGENTS.md`.
- Engineering skill configuration lives in `docs/agents/`.
- Issue tracking is configured for GitHub Issues.
- Triage labels use the default vocabulary.
- Domain documentation is configured as single-context.
- Python project metadata is managed by `uv`.
- A PySide6 desktop app can create and open YOLO Training Projects.
- Active projects can import supported STEM ZC Images into the project queue.
- Imported Normalized Training Images can be annotated with Metal Detection
  Boxes.
- Imported images track explicit review state: `unreviewed`, `labeled`, or
  `reviewed_empty`.
- Reviewed images can be exported as an Ultralytics-compatible train/val
  dataset.
- The GUI can launch the first YOLO fine-tuning workflow from an exported
  dataset, stream logs, show completion/failure status, and cancel a run.
- Training run status and the first run summary are retained in the YOLO
  Training Project folder.
- Completed run summaries identify official `best.pt`, metrics/plots, and the
  fixed dataset used for training.
- Project metadata is persisted in each project folder.
- Smoke, documentation, project-store, image-import, annotation-store,
  dataset-export, training, and GUI workflow tests are present.

The app does not yet support prediction preview or polished metrics dashboards.

## MVS direction

The first MVS is a desktop training tool for rough YOLO detection on STEM ZC
Images. It should help a measurement engineer prepare a reproducible YOLO
Training Project, annotate Metal Detection Boxes, export an Ultralytics-compatible
dataset, fine-tune a pretrained YOLO model, and preview predictions on project
images.

YOLO Trainer is only responsible for training rough detection boxes. It does not
replace Measurer, perform segmentation, define final ROI contracts, or produce
final dimension measurements.

## macOS development

Use macOS for normal development and smoke tests.

```bash
uv sync
uv run pytest
uv run yolo-trainer --smoke
uv run yolo-trainer
```

`uv run yolo-trainer --smoke` starts the PySide6 app, processes one GUI event
cycle, prints a smoke-test message, and exits. This is useful for automated
checks where opening the full GUI is not needed.

## YOLO Training Project workflow

The current GUI supports the first project workflow:

- Create a YOLO Training Project folder.
- Open an existing YOLO Training Project folder.
- Save and load project metadata across app restarts.
- Import supported STEM ZC Images into the active project.
- Show project status and the imported image queue after create/open/import.
- Select an imported image for annotation.
- Draw Metal Detection Boxes by dragging on the Normalized Training Image.
- Autosave annotation labels and restore them after reopening the project.
- Show each image's review state and project-level reviewed/unreviewed progress.
- Mark an image as `reviewed_empty` when it intentionally has no target boxes.
- Export reviewed project images into an Ultralytics-compatible dataset.
- Choose essential fine-tuning settings and start YOLO training from the
  exported dataset.
- Watch streamed training logs while the GUI remains responsive.
- Cancel a running training process and keep the canceled run in project
  history.
- Inspect the latest training run summary, including successful outputs,
  failed-run log context, or canceled status.
- Reject unsupported folders with a clear message in the project view.

Each project folder stores a `yolo-trainer-project.json` metadata file. This is
the current marker used to recognize a valid YOLO Training Project.

Imported images are stored under the project `images/` directory:

- `images/sources/` keeps copied original source files for traceability.
- `images/normalized/` stores 8-bit PNG Normalized Training Images.
- `images/metadata/` stores per-image metadata, including original size,
  normalized size, review state, fixed percentile normalization values, and
  coordinate mapping values.

TIFF import is supported directly. DM3 import is represented by the same public
import workflow with an injectable reader so tests can use a controlled fixture;
hardening a full production DM3 parser is a later step.

Annotation labels are stored under `labels/` as YOLO-compatible `.txt` files.
The canonical class IDs are stable:

| ID  | Class name        |
| --- | ----------------- |
| 0   | `xsection_metal`  |
| 1   | `alongline_metal` |
| 2   | `xsection_via`    |
| 3   | `alongline_via`   |

The GUI may show readable labels, but saved label data should keep these
canonical names and IDs stable for future dataset export.

Review state is stored per imported image. New imports start as `unreviewed`.
Adding one or more Metal Detection Boxes marks the image as `labeled`. Removing
all boxes returns the image to `unreviewed` instead of silently treating it as a
negative training example. The user must explicitly mark an image as
`reviewed_empty` when it contains no target objects.

Dataset export writes an Ultralytics-compatible structure to the selected export
folder:

- `images/train/`
- `images/val/`
- `labels/train/`
- `labels/val/`
- `dataset.yaml`

Only `labeled` and `reviewed_empty` images are exported. `unreviewed` images are
skipped and counted in the GUI export result. Reviewed empty images are exported
as negative examples with empty label files. The first export path uses a fixed
seed and deterministic 80/20 image-level train/val split.

## YOLO fine-tuning workflow

Training starts from the most recently exported dataset for the active YOLO
Training Project. The first GUI training controls expose:

- pretrained model path, default `YOLO11m.pt`
- epochs, default `100`
- image size, default `1024`
- batch, default `auto`
- device, default `0`
- output name, default `yolo-trainer-run`

The GUI launches training in a background process so the window can keep
responding while logs stream into the training log view. Completed, failed, and
canceled run status is written under the project `training/` folder.

The first run summary shows the latest run's status and the `dataset.yaml` used
for training. Completed runs identify official outputs from the Ultralytics run
folder, including `weights/best.pt`, metrics files such as `results.csv`, and
plot images such as `results.png` or `confusion_matrix.png`. Failed runs retain
recent log context for diagnosis. Canceled runs remain visible as canceled, but
partial weights are not presented as official model outputs.

The GUI displays metrics artifacts for inspection only. It does not
automatically assign pass/fail model quality.

Prediction preview, polished metrics dashboards, and model comparison workflows
are later slices.

## Windows training workstation

The expected training workstation is Windows 11 with Python 3.12.8, NVIDIA CUDA
GPU support, and a regular `pip`/`venv` environment.

PyTorch CUDA wheels are intentionally not pinned in this app scaffold. Install
the PyTorch build that matches the workstation driver and CUDA runtime manually,
then install the project dependencies. This avoids locking the repository to one
CUDA wheel that may not match the machine.

The training process imports `ultralytics` inside the active Python environment.
Install Ultralytics in the Windows training environment after the matching
PyTorch CUDA wheel is installed. If Ultralytics is missing, the GUI reports the
training failure instead of changing project data.

Full GPU YOLO training is a manual validation path for the Windows workstation.
Automated tests should not require GPU training.
