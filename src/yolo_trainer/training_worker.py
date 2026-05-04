from __future__ import annotations

import argparse


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolo-trainer-train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Ultralytics is not installed. Install the Windows training "
            "environment dependencies before starting YOLO fine-tuning.",
            flush=True,
        )
        return 1

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=_parse_batch(args.batch),
        device=args.device,
        project=args.project,
        name=args.name,
    )
    return 0


def _parse_batch(value: str) -> int | str:
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
