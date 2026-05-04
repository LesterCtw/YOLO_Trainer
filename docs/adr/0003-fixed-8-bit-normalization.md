# ADR 0003: Use Fixed 8-Bit Normalization

## Status

Accepted

## Context

Source STEM ZC Images can be 16-bit TIFF or DM3 files. YOLO training needs a
stable visual representation, and prediction preview must match training input.

## Decision

Convert source images into Normalized Training Images using a fixed 8-bit
normalization rule. The initial rule is percentile-based normalization using
the 1st and 99th percentiles.

## Consequences

The model sees consistent 8-bit inputs during training and preview. This
reduces sensitivity to extreme outliers. The trade-off is that the fixed rule
may hide some image-specific contrast details, so later slices must keep the
coordinate and normalization metadata testable.
