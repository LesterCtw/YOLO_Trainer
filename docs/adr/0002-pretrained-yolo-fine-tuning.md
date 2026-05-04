# ADR 0002: Fine-Tune A Pretrained YOLO Model

## Status

Accepted

## Context

The MVS needs rough detection of metal and via targets, not a detector trained
from scratch. The expected dataset size is likely too small to justify scratch
training early in the project.

## Decision

Use pretrained YOLO weights as the starting point for training. The default
model for the MVS is YOLO11m, while allowing future compatible Ultralytics
weights to be selected.

## Consequences

Fine-tuning should reach useful results faster and with less data than scratch
training. The trade-off is that model behavior depends on the selected
pretrained weights and the Ultralytics training stack.
