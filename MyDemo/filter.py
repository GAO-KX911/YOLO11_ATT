import argparse
import csv
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser():
    parser = argparse.ArgumentParser(
        description="逐张预测验证集，筛选漏检、小目标和小目标漏检样本。"
    )
    add_arg = parser.add_argument
    add_arg(
        "--weights",
        type=Path,
        default=Path("/home/njust/GAO/V11/V11/runs/detect/train10/weights/best.pt"),
        help="best.pt 路径",
    )
    add_arg(
        "--data",
        type=Path,
        default=Path("/home/njust/GAO/V11/V11/TestData/Data_yaml/D_Fire.yaml"),
        help="数据集 YAML 路径",
    )
    add_arg(
        "--split",
        default="val",
        help="读取数据集 YAML 中的哪个 split，默认 val",
    )
    add_arg(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_val",
        help="输出目录",
    )
    add_arg(
        "--imgsz",
        type=int,
        default=640,
        help="预测尺寸",
    )
    add_arg(
        "--conf",
        type=float,
        default=0.25,
        help="预测置信度阈值",
    )
    add_arg(
        "--iou",
        type=float,
        default=0.5,
        help="GT 与预测框的匹配 IoU 阈值",
    )
    add_arg(
        "--small-thres",
        type=float,
        default=0.0025,
        help="小目标阈值，按归一化面积 w*h 判定",
    )
    add_arg(
        "--device",
        default="0",
        help="推理设备，例如 0、cpu",
    )
    add_arg(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 张图片，0 表示全部处理",
    )
    add_arg(
        "--save-vis",
        action="store_true",
        help="保存带 GT/预测框的可视化图",
    )
    add_arg(
        "--clear-output",
        action="store_true",
        help="运行前清空输出目录",
    )
    return parser


def load_dataset_yaml(data_yaml: Path):
    with data_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_images_dir(cfg: dict, data_yaml: Path, split: str) -> Path:
    if split not in cfg:
        raise KeyError(f"数据集 YAML 中不存在 split={split!r}")

    root = Path(cfg.get("path", ""))
    if root and not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = Path(cfg[split])
    if split_value.is_absolute():
        return split_value
    if root:
        return (root / split_value).resolve()
    return (data_yaml.parent / split_value).resolve()


def resolve_labels_dir(images_dir: Path) -> Path:
    if images_dir.name == "images":
        candidate = images_dir.parent / "labels"
        if candidate.exists():
            return candidate

    candidate = Path(str(images_dir).replace("/images", "/labels"))
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"无法根据图片目录推断标签目录: {images_dir}")


def collect_images(images_dir: Path):
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(image_paths)


def yolo_to_xyxy(xc, yc, w, h, width, height):
    x1 = (xc - w / 2.0) * width
    y1 = (yc - h / 2.0) * height
    x2 = (xc + w / 2.0) * width
    y2 = (yc + h / 2.0) * height
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def load_gt(label_path: Path, width: int, height: int, small_thres: float):
    gts = []
    if not label_path.exists():
        return gts

    with label_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = parts
            try:
                cls_id = int(float(cls))
                xc, yc, w, h = map(float, (xc, yc, w, h))
            except ValueError:
                print(f"跳过非法标签: {label_path}:{line_idx}")
                continue

            area_ratio = w * h
            gts.append(
                {
                    "cls": cls_id,
                    "box": yolo_to_xyxy(xc, yc, w, h, width, height),
                    "area_ratio": area_ratio,
                    "is_small": area_ratio < small_thres,
                }
            )

    return gts


def result_to_preds(result):
    preds = []
    if result.boxes is None or len(result.boxes) == 0:
        return preds

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        preds.append(
            {
                "cls": int(cls_id),
                "conf": float(conf),
                "box": box.tolist(),
            }
        )
    return preds


def match_gts(gts, preds, iou_thres: float):
    missed_indices = []
    for idx, gt in enumerate(gts):
        matched = False
        for pred in preds:
            if pred["cls"] != gt["cls"]:
                continue
            if compute_iou(gt["box"], pred["box"]) >= iou_thres:
                matched = True
                break
        if not matched:
            missed_indices.append(idx)
    return missed_indices


def draw_boxes(image, gts, preds, missed_indices, class_names):
    vis = image.copy()

    for idx, gt in enumerate(gts):
        x1, y1, x2, y2 = [int(round(v)) for v in gt["box"]]
        color = (0, 255, 0)
        thickness = 2
        if idx in missed_indices:
            color = (0, 255, 255)
            thickness = 3
        label = class_names.get(gt["cls"], str(gt["cls"]))
        if gt["is_small"]:
            label = f"{label}|small"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["box"]]
        label = f"{class_names.get(pred['cls'], pred['cls'])}:{pred['conf']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return vis


def ensure_output_dirs(output_root: Path, clear_output: bool):
    if clear_output and output_root.exists():
        shutil.rmtree(output_root)

    dirs = {
        "missed_images": output_root / "missed" / "images",
        "missed_labels": output_root / "missed" / "labels",
        "small_images": output_root / "small" / "images",
        "small_labels": output_root / "small" / "labels",
        "small_missed_images": output_root / "small_missed" / "images",
        "small_missed_labels": output_root / "small_missed" / "labels",
        "visuals": output_root / "visuals",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_sample(image_path: Path, label_path: Path, image_dst: Path, label_dst: Path):
    shutil.copy2(image_path, image_dst / image_path.name)
    if label_path.exists():
        shutil.copy2(label_path, label_dst / label_path.name)


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_dataset_yaml(args.data)
    names = cfg.get("names", {})
    if isinstance(names, list):
        class_names = {idx: name for idx, name in enumerate(names)}
    else:
        class_names = {int(k): v for k, v in names.items()}
    images_dir = resolve_images_dir(cfg, args.data, args.split)
    labels_dir = resolve_labels_dir(images_dir)
    image_paths = collect_images(images_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    output_dirs = ensure_output_dirs(args.output, args.clear_output)
    model = YOLO(str(args.weights))

    summary_rows = []
    stats = {
        "total_images": len(image_paths),
        "missed_images": 0,
        "small_images": 0,
        "small_missed_images": 0,
    }

    print(f"weights: {args.weights}")
    print(f"data yaml: {args.data}")
    print(f"images dir: {images_dir}")
    print(f"labels dir: {labels_dir}")
    print(f"output dir: {args.output}")
    print(f"processing images: {len(image_paths)}")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index}/{len(image_paths)}] 跳过无法读取的图片: {image_path}")
            continue

        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        gts = load_gt(label_path, width, height, args.small_thres)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        preds = result_to_preds(result)
        missed_indices = match_gts(gts, preds, args.iou)

        has_miss = bool(missed_indices)
        small_gt_count = sum(gt["is_small"] for gt in gts)
        small_missed_count = sum(gts[idx]["is_small"] for idx in missed_indices)
        has_small = small_gt_count > 0
        has_small_miss = small_missed_count > 0

        if has_miss:
            copy_sample(
                image_path,
                label_path,
                output_dirs["missed_images"],
                output_dirs["missed_labels"],
            )
            stats["missed_images"] += 1

        if has_small:
            copy_sample(
                image_path,
                label_path,
                output_dirs["small_images"],
                output_dirs["small_labels"],
            )
            stats["small_images"] += 1

        if has_small_miss:
            copy_sample(
                image_path,
                label_path,
                output_dirs["small_missed_images"],
                output_dirs["small_missed_labels"],
            )
            stats["small_missed_images"] += 1

        if args.save_vis and (has_miss or has_small or has_small_miss):
            vis = draw_boxes(image, gts, preds, set(missed_indices), class_names)
            cv2.imwrite(str(output_dirs["visuals"] / image_path.name), vis)

        summary_rows.append(
            {
                "image": image_path.name,
                "label_exists": label_path.exists(),
                "num_gt": len(gts),
                "num_pred": len(preds),
                "num_missed_gt": len(missed_indices),
                "num_small_gt": small_gt_count,
                "num_small_missed_gt": small_missed_count,
                "has_miss": has_miss,
                "has_small": has_small,
                "has_small_miss": has_small_miss,
            }
        )

        if index % 100 == 0 or index == len(image_paths):
            print(
                f"[{index}/{len(image_paths)}] "
                f"missed={stats['missed_images']} "
                f"small={stats['small_images']} "
                f"small_missed={stats['small_missed_images']}"
            )

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "label_exists",
                "num_gt",
                "num_pred",
                "num_missed_gt",
                "num_small_gt",
                "num_small_missed_gt",
                "has_miss",
                "has_small",
                "has_small_miss",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("完成")
    print(f"summary: {summary_path}")
    print(f"missed images: {output_dirs['missed_images']}")
    print(f"small images: {output_dirs['small_images']}")
    print(f"small missed images: {output_dirs['small_missed_images']}")


if __name__ == "__main__":
    main()
