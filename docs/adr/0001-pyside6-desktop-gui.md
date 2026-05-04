# ADR 0001: Use PySide6 Desktop GUI

## Status

Accepted

## Context

YOLO Trainer is used in a measurement workflow where engineers need to load
many local STEM ZC Images, annotate boxes, and launch local training work. The
related Measurer workflow is also desktop-oriented.

## Decision

Build the first application as a PySide6 desktop GUI.

## Consequences

This keeps the workflow local and familiar for measurement engineers. It also
fits macOS development and Windows workstation use. The trade-off is that GUI
smoke tests must account for Qt platform behavior, and the project does not get
browser-based deployment or multi-user collaboration by default.
