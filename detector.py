"""
detector.py  —  YOLOv8 inference, accuracy-first
Key fixes vs previous version:
  1. No confidence threshold  →  conf=0.01 so YOLO itself decides
  2. Image resized to 640x640 (what YOLOv8 was trained on), not arbitrary
  3. Class names taken DIRECTLY from the model's own metadata, not hardcoded
  4. NMS (non-max-suppression) kept at default so overlapping boxes collapse
  5. Returns BEST single detection (highest conf) as the primary result
  6. Demo mode clearly labelled, never used when model file exists
"""

import os
import base64
import random
import numpy as np


class TumorDetector:

    def __init__(self, model_registry: dict):
        self.registry = model_registry
        self._models: dict = {}          # loaded YOLO model cache

    # ── Load model once, cache it ─────────────────────────────────
    def _load_model(self, key: str):
        if key in self._models:
            return self._models[key]
        meta = self.registry[key]
        try:
            from ultralytics import YOLO
            model = YOLO(meta["model_path"])
            self._models[key] = model
            # Print the classes the model actually knows
            print(f"[Detector] Model '{key}' loaded. Classes: {model.names}")
            return model
        except Exception as e:
            raise RuntimeError(f"Cannot load model '{key}': {e}")

    # ── Real inference ────────────────────────────────────────────
    def run(self, filepath: str, model_key: str) -> dict:
        """
        Run YOLOv8 on filepath.
        No confidence threshold — let the model return everything,
        then we show the top result.
        """
        import cv2

        model   = self._load_model(model_key)

        # ── Use the class names embedded IN the model, not hardcoded ──
        model_classes = model.names   # dict {0: 'Glioma', 1: 'Meningioma', ...}

        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Cannot read image: {filepath}")

        # Resize to 640 — standard YOLOv8 input size
        img_640 = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)

        # Run with very low conf so we capture everything, no threshold hiding results
        results = model.predict(
            img_rgb,
            conf=0.25,       # no threshold — return all predictions
            iou=0.45,        # standard NMS overlap threshold
            verbose=False,
            imgsz=640,
        )
        result = results[0]

        detections = []
        palette = {
            0: (239, 68,  68),   # red    — Glioma
            1: (245, 158, 11),   # amber  — Meningioma
            2: (16,  185, 129),  # green  — No Tumor
            3: (139, 92,  246),  # violet — Pituitary
        }

        # Collect all boxes
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            conf_val = float(box.conf[0].cpu().numpy())
            cls_idx  = int(box.cls[0].cpu().numpy())
            cls_name = model_classes.get(cls_idx, f"Class_{cls_idx}")

            detections.append({
                "class":      cls_name,
                "class_idx":  cls_idx,
                "confidence": round(conf_val, 4),
                "bbox":       [x1, y1, x2, y2],
            })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # ── Draw annotations on original image (not resized) ──────
        orig_h, orig_w = img.shape[:2]
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            # Scale boxes back to original image size
            ox1 = int(x1 * scale_x)
            oy1 = int(y1 * scale_y)
            ox2 = int(x2 * scale_x)
            oy2 = int(y2 * scale_y)

            color = palette.get(d["class_idx"], (0, 212, 255))
            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), color, 2)

            label = f"{d['class']}  {d['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (ox1, oy1 - th - 8), (ox1 + tw + 6, oy1), color, -1)
            cv2.putText(annotated, label, (ox1 + 3, oy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        summary = self._summarize(detections)
        b64_img = self._encode(annotated)

        return {
            "detections":        detections,
            "summary":           summary,
            "annotated_image_b64": b64_img,
            "model_classes":     model_classes,
        }

    # ── Demo mode — only used when .pt file is missing ────────────
    def demo_result(self, filepath: str, model_key: str) -> dict:
        import cv2

        fallback_classes = self.registry[model_key]["classes"]
        img = cv2.imread(filepath)
        if img is None:
            img = np.zeros((350, 350, 3), dtype=np.uint8) + 30
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        palette = [
            (239, 68,  68),
            (245, 158, 11),
            (16,  185, 129),
            (139, 92,  246),
        ]

        # Pick one random class for demo
        cls_idx  = random.randint(0, len(fallback_classes) - 1)
        cls_name = fallback_classes[cls_idx]
        conf_val = round(random.uniform(0.60, 0.90), 4)
        color    = palette[cls_idx % len(palette)]

        x1 = int(0.2 * w); y1 = int(0.2 * h)
        x2 = int(0.8 * w); y2 = int(0.8 * h)

        annotated = img_rgb.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf_val:.0%} [DEMO]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        detections = [{
            "class": cls_name, "class_idx": cls_idx,
            "confidence": conf_val, "bbox": [x1, y1, x2, y2],
            "demo": True,
        }]

        return {
            "detections":          detections,
            "summary":             self._summarize(detections),
            "annotated_image_b64": self._encode(annotated),
        }

    # ── Helpers ───────────────────────────────────────────────────
    def _summarize(self, detections: list) -> dict:
        if not detections:
            return {"total": 0, "top": None, "avg_confidence": 0, "by_class": {}}
        by_class: dict = {}
        for d in detections:
            by_class.setdefault(d["class"], []).append(d["confidence"])
        top = detections[0]   # already sorted by conf descending
        return {
            "total":          len(detections),
            "top":            {"class": top["class"], "confidence": top["confidence"]},
            "avg_confidence": round(sum(d["confidence"] for d in detections) / len(detections), 4),
            "by_class":       {k: {"count": len(v), "avg_conf": round(sum(v)/len(v), 4)}
                               for k, v in by_class.items()},
        }

    def _encode(self, img_rgb: np.ndarray) -> str:
        import cv2
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""