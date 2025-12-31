
# validation.py
"""
Ultralytics YOLOv8 validation script for an underwater plastics detection project.

What it does:
1) Loads a trained YOLOv8 model (best.pt by default)
2) Runs validation once and prints/saves core metrics:
   - Precision, Recall, mAP50, mAP50-95
   - Class-wise AP (if multiple classes)
3) Runs a confidence-threshold sweep to help choose the best operating threshold
4) Optional: runs an inference speed (FPS) benchmark using a webcam or a video file
5) Saves results to:
   - outputs/val_summary.json
   - outputs/conf_sweep.csv

Usage examples (Windows):
  python validation.py --data "C:/path/data.yaml" --weights "runs/detect/train/weights/best.pt"
  python validation.py --data "C:/path/data.yaml" --weights "runs/detect/train/weights/best.pt" --conf-sweep
  python validation.py --data "C:/path/data.yaml" --weights "runs/detect/train/weights/best.pt" --speed --source 0
  python validation.py --data "C:/path/data.yaml" --weights "runs/detect/train/weights/best.pt" --speed --source "test.mp4"

Notes:
- This script uses Ultralytics built-in metrics and plotting.
- If you want per-condition tests (clear/murky/low-light), make separate YAMLs or subsets
  and run this script per subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
from ultralytics import YOLO


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def metrics_to_dict(metrics) -> Dict[str, Any]:
    """
    Extracts key metrics from Ultralytics metrics object into a JSON-friendly dict.
    Works across common YOLOv8 versions where fields may differ slightly.
    """
    out: Dict[str, Any] = {}

    # Box-level summary metrics (common fields)
    # Ultralytics typically exposes: metrics.box.p, metrics.box.r, metrics.box.map50, metrics.box.map
    box = getattr(metrics, "box", None)
    if box is not None:
        out["precision"] = safe_float(getattr(box, "p", None))
        out["recall"] = safe_float(getattr(box, "r", None))
        out["map50"] = safe_float(getattr(box, "map50", None))
        out["map50_95"] = safe_float(getattr(box, "map", None))
        # Some versions also have f1
        out["f1"] = safe_float(getattr(box, "f1", None))

        # Class-wise AP arrays (if present)
        # "ap" can be per-class AP50-95; "ap50" per-class AP50
        ap = getattr(box, "ap", None)
        ap50 = getattr(box, "ap50", None)

        if ap is not None:
            try:
                out["ap_per_class_map50_95"] = [safe_float(v) for v in list(ap)]
            except Exception:
                pass
        if ap50 is not None:
            try:
                out["ap50_per_class"] = [safe_float(v) for v in list(ap50)]
            except Exception:
                pass

    # Some Ultralytics versions provide "names" on metrics, others on model
    names = getattr(metrics, "names", None)
    if names is not None:
        out["class_names"] = names

    return out


def run_validation(
    weights: str,
    data: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.6,
    device: Optional[str] = None,
    plots: bool = True,
    save_json: bool = True,
    project: str = "runs/validate",
    name: str = "val",
    verbose: bool = True,
) -> Dict[str, Any]:
    model = YOLO(weights)

    # Ultralytics val() will create a run folder under project/name
    kwargs = dict(
        data=data,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=plots,
        save_json=save_json,
        project=project,
        name=name,
        verbose=verbose,
    )
    if device:
        kwargs["device"] = device

    metrics = model.val(**kwargs)

    # Build summary dict
    summary = metrics_to_dict(metrics)
    summary.update(
        {
            "weights": weights,
            "data": data,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "device": device,
            "project": project,
            "name": name,
        }
    )

    # If class names missing, pull from model where possible
    if "class_names" not in summary or summary["class_names"] is None:
        try:
            summary["class_names"] = model.model.names  # type: ignore[attr-defined]
        except Exception:
            pass

    return summary


def compute_f1(p: float, r: float) -> Optional[float]:
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return (2 * p * r) / (p + r)


def confidence_sweep(
    weights: str,
    data: str,
    imgsz: int,
    iou: float,
    conf_values: List[float],
    device: Optional[str],
    project: str,
    base_name: str,
    outputs_dir: Path,
    verbose: bool,
) -> Path:
    """
    Runs val() for multiple confidence thresholds and writes a CSV:
    conf, precision, recall, f1, map50, map50_95
    """
    csv_path = outputs_dir / "conf_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conf", "precision", "recall", "f1", "map50", "map50_95"]
        )
        writer.writeheader()

        for t in conf_values:
            name = f"{base_name}_conf{str(t).replace('.', '_')}"
            summary = run_validation(
                weights=weights,
                data=data,
                imgsz=imgsz,
                conf=t,
                iou=iou,
                device=device,
                plots=False,         # keep sweep lighter
                save_json=False,     # keep sweep lighter
                project=project,
                name=name,
                verbose=verbose,
            )

            p = summary.get("precision")
            r = summary.get("recall")
            f1 = summary.get("f1")
            if f1 is None and p is not None and r is not None:
                f1 = compute_f1(p, r)

            row = {
                "conf": t,
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": summary.get("map50"),
                "map50_95": summary.get("map50_95"),
            }
            writer.writerow(row)

    return csv_path


def speed_test(
    weights: str,
    source: Union[int, str],
    imgsz: int,
    conf: float,
    n_frames: int,
    device: Optional[str],
    show: bool = False,
) -> Dict[str, Any]:
    """
    Benchmarks inference speed (FPS) on webcam (source=0/1/2) or a video file path.
    """
    model = YOLO(weights)

    # OpenCV capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # Warm-up
    warmup = min(10, n_frames)
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            break
        _ = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)

    # Timed loop
    processed = 0
    start = time.time()
    while processed < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
        processed += 1

        if show:
            # Draw boxes for visualization
            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Speed Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    end = time.time()
    cap.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = max(1e-6, end - start)
    fps = processed / elapsed

    return {
        "weights": weights,
        "source": source,
        "imgsz": imgsz,
        "conf": conf,
        "frames_processed": processed,
        "elapsed_seconds": elapsed,
        "fps": fps,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 validation metrics runner")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to trained weights (best.pt recommended)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for val/predict")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for main val")
    parser.add_argument("--iou", type=float, default=0.60, help="IoU threshold for mAP calculation")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device like "cpu" or "0" (GPU id). If omitted, Ultralytics auto-selects.',
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/validate",
        help="Ultralytics project directory for val runs",
    )
    parser.add_argument("--name", type=str, default="val", help="Run name under project/")
    parser.add_argument("--no-plots", action="store_true", help="Disable PR/confusion plots")
    parser.add_argument("--no-save-json", action="store_true", help="Disable saving COCO-style JSON")
    parser.add_argument("--quiet", action="store_true", help="Less console output")

    # Confidence sweep
    parser.add_argument(
        "--conf-sweep",
        action="store_true",
        help="Run validation across multiple confidence thresholds and save CSV",
    )
    parser.add_argument(
        "--conf-values",
        type=str,
        default="0.2,0.3,0.4,0.5,0.6",
        help="Comma-separated confidence thresholds for sweep (e.g., 0.1,0.2,0.3)",
    )

    # Speed test
    parser.add_argument("--speed", action="store_true", help="Run inference FPS benchmark")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Video source: "0" for webcam, or file path like "test.mp4"',
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of frames for speed test")
    parser.add_argument("--show", action="store_true", help="Show annotated frames (press q to quit)")

    # Output dir for summary files
    parser.add_argument("--outdir", type=str, default="outputs", help="Where to save summary JSON/CSV")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = args.data
    weights = args.weights
    imgsz = args.imgsz
    conf = args.conf
    iou = args.iou
    device = args.device
    plots = not args.no_plots
    save_json = not args.no_save_json
    verbose = not args.quiet

    outputs_dir = ensure_dir(args.outdir)

    # 1) Main validation
    summary = run_validation(
        weights=weights,
        data=data,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        plots=plots,
        save_json=save_json,
        project=args.project,
        name=args.name,
        verbose=verbose,
    )

    # Compute F1 if missing
    if summary.get("f1") is None and summary.get("precision") is not None and summary.get("recall") is not None:
        summary["f1"] = compute_f1(summary["precision"], summary["recall"])

    # Save JSON summary
    summary_path = outputs_dir / "val_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Validation Summary ===")
    print(f"Weights: {summary.get('weights')}")
    print(f"Data:    {summary.get('data')}")
    print(f"imgsz:   {summary.get('imgsz')}, conf: {summary.get('conf')}, iou: {summary.get('iou')}")
    print(f"Precision:  {summary.get('precision')}")
    print(f"Recall:     {summary.get('recall')}")
    print(f"F1:         {summary.get('f1')}")
    print(f"mAP50:      {summary.get('map50')}")
    print(f"mAP50-95:   {summary.get('map50_95')}")
    print(f"Saved JSON: {summary_path}")

    # 2) Confidence sweep (optional)
    if args.conf_sweep:
        conf_values = []
        for part in args.conf_values.split(","):
            part = part.strip()
            if not part:
                continue
            conf_values.append(float(part))

        csv_path = confidence_sweep(
            weights=weights,
            data=data,
            imgsz=imgsz,
            iou=iou,
            conf_values=conf_values,
            device=device,
            project=args.project,
            base_name=args.name,
            outputs_dir=outputs_dir,
            verbose=verbose,
        )
        print(f"\nSaved confidence sweep CSV: {csv_path}")

    # 3) Speed test (optional)
    if args.speed:
        # interpret source: if it's a digit string, treat as webcam index
        src: Union[int, str]
        if args.source.isdigit():
            src = int(args.source)
        else:
            src = args.source

        speed = speed_test(
            weights=weights,
            source=src,
            imgsz=imgsz,
            conf=conf,
            n_frames=args.frames,
            device=device,
            show=args.show,
        )

        speed_path = outputs_dir / "speed_test.json"
        with speed_path.open("w", encoding="utf-8") as f:
            json.dump(speed, f, indent=2)

        print("\n=== Speed Test ===")
        print(f"Source: {speed['source']}")
        print(f"Frames processed: {speed['frames_processed']}")
        print(f"Elapsed (s): {speed['elapsed_seconds']:.3f}")
        print(f"FPS @ {imgsz}: {speed['fps']:.2f}")
        print(f"Saved speed JSON: {speed_path}")


if __name__ == "__main__":
    main()

