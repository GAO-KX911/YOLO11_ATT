# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (Dict[str, Any], optional): Arguments for the validator.
            _callbacks (List[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
        self.metrics_small_images = DetMetrics()
        self.metrics_small = DetMetrics()

        # Added: model parameter caches
        self.model_params = 0
        self.model_trainable_params = 0
        self.model_params_m = 0.0

        # Added: small-object threshold (normalized bbox area in YOLO format)
        # Default 0.0025 ~= (32 / 640)^2
        self.small_area_thres = float(getattr(self.args, "small_area_thres", 0.0025))
        self.small_seen = 0
        self.small_image_seen = 0

    @staticmethod
    def _raw_area_ratio_from_batch_boxes(
        bbox_xywh: torch.Tensor, ori_shape: Tuple[int, int], imgsz: Tuple[int, int], ratio_pad: Any
    ) -> torch.Tensor:
        """Recover normalized box areas in the original image space from letterboxed batch boxes."""
        if len(bbox_xywh) == 0:
            return torch.zeros(0, device=bbox_xywh.device)

        ori_h, ori_w = ori_shape
        img_h, img_w = imgsz
        if isinstance(ratio_pad, (tuple, list)) and len(ratio_pad) >= 1:
            ratio = ratio_pad[0]
            if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
                ratio_w, ratio_h = float(ratio[0]), float(ratio[1])
            else:
                ratio_w = ratio_h = float(ratio)
        else:
            # Fallback for unexpected formats.
            ratio_w = img_w / max(ori_w, 1e-9)
            ratio_h = img_h / max(ori_h, 1e-9)

        raw_w = bbox_xywh[:, 2] * img_w / max(ori_w * ratio_w, 1e-9)
        raw_h = bbox_xywh[:, 3] * img_h / max(ori_h * ratio_h, 1e-9)
        return raw_w * raw_h

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.

        Returns:
            (Dict[str, Any]): Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in {"batch_idx", "cls", "bboxes"}:
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.small_seen = 0
        self.small_image_seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.metrics_small_images.names = model.names
        self.metrics_small.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

        # Added: collect model params
        self.model_params = int(sum(p.numel() for p in model.parameters()))
        self.model_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.model_params_m = self.model_params / 1e6

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions after NMS.
        """
        outputs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox_xywh = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]

        if len(cls):
            raw_area_ratio = self._raw_area_ratio_from_batch_boxes(bbox_xywh, ori_shape, imgsz, ratio_pad)
            small_mask = raw_area_ratio < self.small_area_thres
            bbox = ops.xywh2xyxy(bbox_xywh) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        else:
            raw_area_ratio = torch.zeros(0, device=self.device)
            small_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            bbox = bbox_xywh

        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
            "small_mask": small_mask,
            "raw_area_ratio": raw_area_ratio,
        }

    def _prepare_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            # Added: small-object metrics (image subset containing at least one small GT)
            small_mask = pbatch["small_mask"]
            if len(small_mask) and small_mask.any():
                self.small_image_seen += 1
                self.metrics_small_images.update_stats(
                    {
                        **self._process_batch(predn, pbatch),
                        "target_cls": cls,
                        "target_img": np.unique(cls),
                        "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                        "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    }
                )

                self.small_seen += 1
                small_cls_t = pbatch["cls"][small_mask]
                small_boxes_t = pbatch["bboxes"][small_mask]
                small_cls = small_cls_t.cpu().numpy()
                self.metrics_small.update_stats(
                    {
                        **self._process_batch(predn, {"cls": small_cls_t, "bboxes": small_boxes_t}),
                        "target_cls": small_cls,
                        "target_img": np.unique(small_cls),
                        "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                        "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    }
                )

            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir
        self.metrics_small_images.speed = self.speed
        self.metrics_small_images.save_dir = self.save_dir
        self.metrics_small.speed = self.speed
        self.metrics_small.save_dir = self.save_dir

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate and return metrics statistics.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        if len(getattr(self.metrics_small_images, "stats", [])):
            self.metrics_small_images.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
        if len(getattr(self.metrics_small, "stats", [])):
            self.metrics_small.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)

        results = self.metrics.results_dict.copy()
        if len(getattr(self.metrics_small_images, "stats", [])):
            small_image_mean = self.metrics_small_images.mean_results()
            results["small_image/P"] = small_image_mean[0]
            results["small_image/R"] = small_image_mean[1]
            results["small_image/mAP50"] = small_image_mean[2]
            results["small_image/mAP50-95"] = small_image_mean[3]
        if len(getattr(self.metrics_small, "stats", [])):
            small_mean = self.metrics_small.mean_results()
            results["small/P"] = small_mean[0]
            results["small/R"] = small_mean[1]
            results["small/mAP50"] = small_mean[2]
            results["small/mAP50-95"] = small_mean[3]

        self.metrics.clear_stats()
        self.metrics_small_images.clear_stats()
        self.metrics_small.clear_stats()
        return results

    def save_metrics_to_file(self) -> None:
        """Save validation metrics, small-object metrics, speed, and model params to json/txt file."""
        json_path = self.save_dir / "metrics_summary.json"
        txt_path = self.save_dir / "metrics_summary.txt"

        mean_res = self.metrics.mean_results()  # [P, R, mAP50, mAP50-95]

        speed_data = {}
        if hasattr(self, "speed") and isinstance(self.speed, dict):
            speed_data = {k: float(v) for k, v in self.speed.items()}
        elif hasattr(self.metrics, "speed") and isinstance(self.metrics.speed, dict):
            speed_data = {k: float(v) for k, v in self.metrics.speed.items()}

        total_speed = float(sum(speed_data.values())) if speed_data else 0.0

        nt_per_class = getattr(self.metrics, "nt_per_class", np.zeros(self.nc, dtype=int))
        nt_per_image = getattr(self.metrics, "nt_per_image", np.zeros(self.nc, dtype=int))
        ap_class_index = getattr(self.metrics, "ap_class_index", [])
        stats_len = len(getattr(self.metrics, "stats", []))

        small_image_stats_len = len(getattr(self.metrics_small_images, "stats", []))
        small_image_nt_per_class = getattr(self.metrics_small_images, "nt_per_class", np.zeros(self.nc, dtype=int))
        small_image_nt_per_image = getattr(self.metrics_small_images, "nt_per_image", np.zeros(self.nc, dtype=int))
        small_image_ap_class_index = getattr(self.metrics_small_images, "ap_class_index", [])

        small_stats_len = len(getattr(self.metrics_small, "stats", []))
        small_nt_per_class = getattr(self.metrics_small, "nt_per_class", np.zeros(self.nc, dtype=int))
        small_nt_per_image = getattr(self.metrics_small, "nt_per_image", np.zeros(self.nc, dtype=int))
        small_ap_class_index = getattr(self.metrics_small, "ap_class_index", [])

        data = {
            "model": {
                "params": int(self.model_params),
                "trainable_params": int(self.model_trainable_params),
                "params_M": float(self.model_params_m),
            },
            "small_object_rule": {
                "area_threshold": float(self.small_area_thres),
                "definition": "bbox_w * bbox_h < threshold (normalized YOLO area from raw labels)",
            },
            "speed_ms_per_image": {
                **speed_data,
                "total": total_speed,
            },
            "metric_definitions": {
                "small_image_all": "metrics on all GT/predictions from images containing at least one small GT",
                "small_all": "metrics on small GT instances only",
            },
            "all": {
                "images": int(self.seen),
                "instances": int(nt_per_class.sum()) if nt_per_class is not None else 0,
                "P": float(mean_res[0]),
                "R": float(mean_res[1]),
                "mAP50": float(mean_res[2]),
                "mAP50_95": float(mean_res[3]),
            },
            "classes": {},
            "small_image_all": None,
            "small_image_classes": {},
            "small_all": None,
            "small_classes": {},
        }

        # per-class metrics
        if self.nc > 1 and stats_len:
            for i, c in enumerate(ap_class_index):
                cls_res = self.metrics.class_result(i)  # [P, R, mAP50, mAP50-95]
                data["classes"][self.names[c]] = {
                    "images": int(nt_per_image[c]),
                    "instances": int(nt_per_class[c]),
                    "P": float(cls_res[0]),
                    "R": float(cls_res[1]),
                    "mAP50": float(cls_res[2]),
                    "mAP50_95": float(cls_res[3]),
                }

        # metrics on the whole subset of images that contain at least one small GT
        if self.nc > 1 and small_image_stats_len:
            small_image_mean = self.metrics_small_images.mean_results()
            data["small_image_all"] = {
                "images": int(self.small_image_seen),
                "instances": int(small_image_nt_per_class.sum()) if small_image_nt_per_class is not None else 0,
                "P": float(small_image_mean[0]),
                "R": float(small_image_mean[1]),
                "mAP50": float(small_image_mean[2]),
                "mAP50_95": float(small_image_mean[3]),
            }
            for i, c in enumerate(small_image_ap_class_index):
                cls_res = self.metrics_small_images.class_result(i)
                data["small_image_classes"][self.names[c]] = {
                    "images": int(small_image_nt_per_image[c]),
                    "instances": int(small_image_nt_per_class[c]),
                    "P": float(cls_res[0]),
                    "R": float(cls_res[1]),
                    "mAP50": float(cls_res[2]),
                    "mAP50_95": float(cls_res[3]),
                }

        # small-object metrics
        if self.nc > 1 and small_stats_len:
            small_mean = self.metrics_small.mean_results()
            data["small_all"] = {
                "images": int(self.small_seen),
                "instances": int(small_nt_per_class.sum()) if small_nt_per_class is not None else 0,
                "P": float(small_mean[0]),
                "R": float(small_mean[1]),
                "mAP50": float(small_mean[2]),
                "mAP50_95": float(small_mean[3]),
            }
            for i, c in enumerate(small_ap_class_index):
                cls_res = self.metrics_small.class_result(i)
                data["small_classes"][self.names[c]] = {
                    "images": int(small_nt_per_image[c]),
                    "instances": int(small_nt_per_class[c]),
                    "P": float(cls_res[0]),
                    "R": float(cls_res[1]),
                    "mAP50": float(cls_res[2]),
                    "mAP50_95": float(cls_res[3]),
                }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== Validation Metrics Summary ===\n\n")

            f.write("[Model]\n")
            f.write(f"params={data['model']['params']}\n")
            f.write(f"trainable_params={data['model']['trainable_params']}\n")
            f.write(f"params_M={data['model']['params_M']:.6f}\n\n")

            f.write("[Small Object Rule]\n")
            f.write(f"area_threshold={data['small_object_rule']['area_threshold']:.6f}\n")
            f.write(f"definition={data['small_object_rule']['definition']}\n\n")

            f.write("[Metric Definitions]\n")
            f.write(f"small_image_all={data['metric_definitions']['small_image_all']}\n")
            f.write(f"small_all={data['metric_definitions']['small_all']}\n\n")

            f.write("[Speed] (ms/image)\n")
            for k, v in data["speed_ms_per_image"].items():
                f.write(f"{k}={v:.6f}\n")
            f.write("\n")

            f.write("[All]\n")
            f.write(f"images={data['all']['images']}\n")
            f.write(f"instances={data['all']['instances']}\n")
            f.write(f"P={data['all']['P']:.6f}\n")
            f.write(f"R={data['all']['R']:.6f}\n")
            f.write(f"mAP50={data['all']['mAP50']:.6f}\n")
            f.write(f"mAP50_95={data['all']['mAP50_95']:.6f}\n\n")

            if data["classes"]:
                f.write("[Per-class]\n")
                for name, vals in data["classes"].items():
                    f.write(
                        f"{name}: images={vals['images']}, "
                        f"instances={vals['instances']}, "
                        f"P={vals['P']:.6f}, "
                        f"R={vals['R']:.6f}, "
                        f"mAP50={vals['mAP50']:.6f}, "
                        f"mAP50_95={vals['mAP50_95']:.6f}\n"
                    )
                f.write("\n")

            if data["small_image_all"] is not None:
                f.write("[Small-Image-All]\n")
                f.write(f"images={data['small_image_all']['images']}\n")
                f.write(f"instances={data['small_image_all']['instances']}\n")
                f.write(f"P={data['small_image_all']['P']:.6f}\n")
                f.write(f"R={data['small_image_all']['R']:.6f}\n")
                f.write(f"small_image_mAP50={data['small_image_all']['mAP50']:.6f}\n")
                f.write(f"small_image_mAP50_95={data['small_image_all']['mAP50_95']:.6f}\n\n")

            if data["small_image_classes"]:
                f.write("[Small-Image-Per-class]\n")
                for name, vals in data["small_image_classes"].items():
                    f.write(
                        f"{name}: images={vals['images']}, "
                        f"instances={vals['instances']}, "
                        f"P={vals['P']:.6f}, "
                        f"R={vals['R']:.6f}, "
                        f"small_image_mAP50={vals['mAP50']:.6f}, "
                        f"small_image_mAP50_95={vals['mAP50_95']:.6f}\n"
                    )
                f.write("\n")

            if data["small_all"] is not None:
                f.write("[Small-All]\n")
                f.write(f"images={data['small_all']['images']}\n")
                f.write(f"instances={data['small_all']['instances']}\n")
                f.write(f"P={data['small_all']['P']:.6f}\n")
                f.write(f"R={data['small_all']['R']:.6f}\n")
                f.write(f"small_mAP50={data['small_all']['mAP50']:.6f}\n")
                f.write(f"small_mAP50_95={data['small_all']['mAP50_95']:.6f}\n\n")

            if data["small_classes"]:
                f.write("[Small-Per-class]\n")
                for name, vals in data["small_classes"].items():
                    f.write(
                        f"{name}: images={vals['images']}, "
                        f"instances={vals['instances']}, "
                        f"P={vals['P']:.6f}, "
                        f"R={vals['R']:.6f}, "
                        f"small_mAP50={vals['mAP50']:.6f}, "
                        f"small_mAP50_95={vals['mAP50_95']:.6f}\n"
                    )

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

        # Added: print whole small-image subset summary if available
        if self.nc > 1 and len(getattr(self.metrics_small_images, "stats", [])):
            small_image_mean = self.metrics_small_images.mean_results()
            LOGGER.info(
                "[small-img]".rjust(22)
                + f"{self.small_image_seen:11d}{int(self.metrics_small_images.nt_per_class.sum()):11d}"
                + "".join(f"{x:11.3g}" for x in small_image_mean)
            )

        # Added: print small-GT summary if available
        if self.nc > 1 and len(getattr(self.metrics_small, "stats", [])):
            small_mean = self.metrics_small.mean_results()
            LOGGER.info(
                "[small-gt]".rjust(22)
                + f"{self.small_seen:11d}{int(self.metrics_small.nt_per_class.sum()):11d}"
                + "".join(f"{x:11.3g}" for x in small_mean)
            )

        # Added: save metrics, params, speed, small metrics
        self.save_metrics_to_file()

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int, max_det: Optional[int] = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.
        """
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        Serialize YOLO predictions to COCO json format.
        """
        stem = Path(pbatch["im_file"]).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: Dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: Union[str, List[str]] = "bbox",
        suffix: Union[str, List[str]] = "Box",
    ) -> Dict[str, Any]:
        """
        Evaluate COCO/LVIS metrics using faster-coco-eval library.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
'''
from pathlib import Path
path = Path('/mnt/data/DetectionValidator_modified_with_small.py')
path.write_text(modified, encoding='utf-8')
print(path)
print(path.exists(), path.stat().st_size)
'''
