from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataEngine:
    """Generate structured metadata and urgency information from detections."""

    URGENCY_RULES = {
        "Glioma": "High",
        "Meningioma": "Moderate",
        "Pituitary": "Moderate",
        "No Tumor": "Routine",
    }

    RECOMMENDATIONS = {
        "High": "Prompt review by a qualified radiologist or neurospecialist is recommended.",
        "Moderate": "Specialist consultation and clinical correlation are recommended.",
        "Routine": "No tumor was detected by the model. Routine clinical review is still recommended.",
    }

    def generate(
        self,
        image_path: str,
        detections: list[dict[str, Any]],
        model_label: str,
        elapsed_ms: int,
    ) -> dict[str, Any]:
        image = Path(image_path)

        primary_detection = self._get_primary_detection(detections)
        finding = primary_detection.get("class", "No Tumor")
        confidence = round(float(primary_detection.get("confidence", 0)) * 100, 2)

        urgency = self.URGENCY_RULES.get(finding, "Moderate")

        return {
            "scan_id": image.stem,
            "filename": image.name,
            "modality": "Brain MRI",
            "analysis_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "model": model_label,
            "finding": finding,
            "confidence_percent": confidence,
            "detection_count": len(detections),
            "urgency": urgency,
            "recommendation": self.RECOMMENDATIONS[urgency],
            "processing_time_ms": elapsed_ms,
            "bounding_box": primary_detection.get("bbox"),
        }

    @staticmethod
    def _get_primary_detection(
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not detections:
            return {
                "class": "No Tumor",
                "confidence": 0,
                "bbox": None,
            }

        return max(
            detections,
            key=lambda item: float(item.get("confidence", 0)),
        )