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
- A PySide6 desktop app can create and open empty YOLO Training Projects.
- Project metadata is persisted in each project folder.
- Smoke, documentation, project-store, and GUI project workflow tests are present.

The app does not yet support image import, annotation, dataset export, YOLO
training, or prediction preview.

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

The current GUI supports the first empty-project workflow:

- Create a YOLO Training Project folder.
- Open an existing YOLO Training Project folder.
- Save and load project metadata across app restarts.
- Show project status and an empty image queue after create/open.
- Reject unsupported folders with a clear message in the project view.

Each project folder stores a `yolo-trainer-project.json` metadata file. This is
the current marker used to recognize a valid YOLO Training Project.

## Windows training workstation

The expected training workstation is Windows 11 with Python 3.12.8, NVIDIA CUDA
GPU support, and a regular `pip`/`venv` environment.

PyTorch CUDA wheels are intentionally not pinned in this app scaffold. Install
the PyTorch build that matches the workstation driver and CUDA runtime manually,
then install the project dependencies. This avoids locking the repository to one
CUDA wheel that may not match the machine.

Full GPU YOLO training is a manual validation path for the Windows workstation.
Automated tests should not require GPU training.
