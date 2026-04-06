from __future__ import annotations

import argparse
import random
from pathlib import Path


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an oversampled training image list for a target YOLO class.")
    parser.add_argument("--img-dir", type=Path, required=True, help="Directory containing training images.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Directory containing YOLO label txt files.")
    parser.add_argument("--out", type=Path, required=True, help="Output txt path.")
    parser.add_argument("--class-id", type=int, default=1, help="Target class id to oversample.")
    parser.add_argument("--repeat-factor", type=float, default=1.5, help="Repeat factor for matching images.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for fractional repeats.")
    return parser.parse_args()


def has_target_class(label_path: Path, class_id: int) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts and int(float(parts[0])) == class_id:
            return True
    return False


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.repeat_factor < 1.0:
        raise ValueError("--repeat-factor must be >= 1.0")

    image_paths = sorted(p for p in args.img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    whole_repeats = int(args.repeat_factor) - 1
    fractional_repeat = args.repeat_factor - int(args.repeat_factor)

    lines = []
    total_target_images = 0
    extra_copies = 0

    for img_path in image_paths:
        label_path = args.label_dir / f"{img_path.stem}.txt"
        lines.append(str(img_path))

        if not has_target_class(label_path, args.class_id):
            continue

        total_target_images += 1
        for _ in range(whole_repeats):
            lines.append(str(img_path))
            extra_copies += 1

        if fractional_repeat > 0 and rng.random() < fractional_repeat:
            lines.append(str(img_path))
            extra_copies += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"images={len(image_paths)}")
    print(f"target_images={total_target_images}")
    print(f"repeat_factor={args.repeat_factor}")
    print(f"extra_copies={extra_copies}")
    print(f"total_entries={len(lines)}")
    print(f"saved_to={args.out}")


if __name__ == "__main__":
    main()
