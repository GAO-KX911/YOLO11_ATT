from __future__ import annotations

import argparse
import random
from pathlib import Path


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an oversampled training image list using joint fire/small-object rules."
    )
    parser.add_argument("--img-dir", type=Path, required=True, help="Directory containing training images.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Directory containing YOLO label txt files.")
    parser.add_argument("--out", type=Path, required=True, help="Output txt path.")
    parser.add_argument("--fire-class-id", type=int, default=1, help="Class id for fire.")
    parser.add_argument(
        "--small-area-thres",
        type=float,
        default=0.0025,
        help="Small-object threshold using normalized YOLO area w*h.",
    )
    parser.add_argument("--fire-factor", type=float, default=1.3, help="Repeat factor for images containing fire.")
    parser.add_argument(
        "--small-factor", type=float, default=1.3, help="Repeat factor for images containing at least one small GT."
    )
    parser.add_argument(
        "--both-factor",
        type=float,
        default=1.8,
        help="Repeat factor for images containing both fire and small GT.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for fractional repeats.")
    return parser.parse_args()


def get_image_flags(label_path: Path, fire_class_id: int, small_area_thres: float) -> tuple[bool, bool]:
    has_fire = False
    has_small = False
    if not label_path.exists():
        return has_fire, has_small

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        w = float(parts[3])
        h = float(parts[4])
        if cls_id == fire_class_id:
            has_fire = True
        if w * h < small_area_thres:
            has_small = True
        if has_fire and has_small:
            break
    return has_fire, has_small


def choose_repeat_factor(has_fire: bool, has_small: bool, fire_factor: float, small_factor: float, both_factor: float) -> float:
    if has_fire and has_small:
        return both_factor
    if has_fire:
        return fire_factor
    if has_small:
        return small_factor
    return 1.0


def append_with_factor(lines: list[str], path: str, factor: float, rng: random.Random) -> int:
    whole_repeats = int(factor)
    fractional_repeat = factor - whole_repeats
    added = 0
    for _ in range(whole_repeats):
        lines.append(path)
        added += 1
    if fractional_repeat > 0 and rng.random() < fractional_repeat:
        lines.append(path)
        added += 1
    return added


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    for name, value in (
        ("fire-factor", args.fire_factor),
        ("small-factor", args.small_factor),
        ("both-factor", args.both_factor),
    ):
        if value < 1.0:
            raise ValueError(f"--{name} must be >= 1.0")

    image_paths = sorted(p for p in args.img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    lines: list[str] = []

    fire_images = 0
    small_images = 0
    both_images = 0
    extra_copies = 0

    for img_path in image_paths:
        label_path = args.label_dir / f"{img_path.stem}.txt"
        has_fire, has_small = get_image_flags(label_path, args.fire_class_id, args.small_area_thres)
        if has_fire:
            fire_images += 1
        if has_small:
            small_images += 1
        if has_fire and has_small:
            both_images += 1

        factor = choose_repeat_factor(has_fire, has_small, args.fire_factor, args.small_factor, args.both_factor)
        added = append_with_factor(lines, str(img_path), factor, rng)
        extra_copies += added - 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"images={len(image_paths)}")
    print(f"fire_images={fire_images}")
    print(f"small_images={small_images}")
    print(f"both_images={both_images}")
    print(f"fire_factor={args.fire_factor}")
    print(f"small_factor={args.small_factor}")
    print(f"both_factor={args.both_factor}")
    print(f"extra_copies={extra_copies}")
    print(f"total_entries={len(lines)}")
    print(f"saved_to={args.out}")


if __name__ == "__main__":
    main()
