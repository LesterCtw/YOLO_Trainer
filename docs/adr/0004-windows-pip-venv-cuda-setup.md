# ADR 0004: Use Windows pip/venv CUDA Setup For Training

## Status

Accepted

## Context

Normal development happens on macOS with `uv`, but practical YOLO training is
expected to happen on a Windows 11 workstation with an NVIDIA GPU.

## Decision

Document Windows training as a Python 3.12.8 `pip`/`venv` setup. Install the
PyTorch CUDA wheel manually for the workstation instead of pinning a
CUDA-specific torch wheel in the project dependencies.

## Consequences

This keeps the repository usable across macOS development and Windows training
machines. It avoids forcing one CUDA wheel onto every environment. The trade-off
is that Windows GPU setup has a manual step and must be validated on the actual
training workstation.
