"""
PCB Component Localization and Automated Inventory Counting Pipeline
Author: Verina Fouad Farid Khalil
Role: Detection Architecture & Counting Pipeline Lead

Leverages YOLOv8 for dense object detection across populated PCBs,
bounding 8,337 annotated component instances across 22 classes.
"""

from collections import Counter
from typing import Dict, Any, List

class PCBComponentDetector:
    """Performs inference and generates automated bill-of-materials (BOM) inventories."""
    def __init__(self, model_weights: str = "weights/yolov8n_pcb_best.pt", conf_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.model_weights = model_weights
        # In production, load via: from ultralytics import YOLO; self.model = YOLO(model_weights)

    def parse_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parses bounding boxes, scores, and tabulates automated component inventory count.
        """
        inventory = Counter()
        annotated_results = []

        for det in detections:
            if det["confidence"] >= self.conf_threshold:
                cls_name = det["class_name"]
                inventory[cls_name] += 1
                annotated_results.append({
                    "bbox": det["bbox"],  # [xmin, ymin, xmax, ymax]
                    "class": cls_name,
                    "confidence": round(det["confidence"], 3)
                })

        return {
            "total_detected": sum(inventory.values()),
            "inventory_summary": dict(inventory),
            "detections": annotated_results
        }
