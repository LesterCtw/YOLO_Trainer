# YOLO Trainer Context

This glossary records the project language used by issues, tests, and future
implementation slices.

## Terms

### STEM ZC Image

A microscopy source image used as YOLO Trainer input. In the MVS, these images
come from TIFF or DM3 files and may contain MOM, BEOL, metal, or via structures.
The original source file must remain traceable because downstream work may need
coordinates mapped back to the original image.

### Metal Detection Box

A rough YOLO bounding box around a visible metal or via target. It is training
data for detection, not a precise ROI, segmentation mask, refined boundary, or
final measurement region.

### YOLO Training Project

The local project workspace that groups source images, normalized training
images, labels, review state, dataset exports, training runs, and prediction
preview outputs for one training effort.

### Normalized Training Image

An 8-bit image derived from a source STEM ZC Image using the fixed normalization
rule. YOLO training and prediction preview should use this representation so
the model sees stable image inputs.

### Reviewed Empty Image

An image that a user has intentionally reviewed and marked as containing no
target object. It can be included as a negative training example. This is
different from an unreviewed image, which must not be treated as ready for
training.
