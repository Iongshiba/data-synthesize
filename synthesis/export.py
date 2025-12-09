"""
Annotation export utilities for COCO and YOLO formats
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class COCOExporter:
    """Export annotations in COCO format"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotations = {
            "info": {
                "description": "Synthetic dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        self.image_id = 0
        self.annotation_id = 0
        self.category_map = {}

    def add_category(self, category_id: int, category_name: str):
        """Add a category to the dataset"""
        if category_id not in self.category_map:
            self.annotations["categories"].append(
                {"id": category_id, "name": category_name, "supercategory": "object"}
            )
            self.category_map[category_id] = category_name

    def add_image(self, file_name: str, width: int, height: int) -> int:
        """
        Add an image to the dataset

        Returns:
            image_id
        """
        image_id = self.image_id
        self.image_id += 1

        self.annotations["images"].append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

        return image_id

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: Tuple[int, int, int, int],
        segmentation_mask: np.ndarray = None,
        area: int = None,
    ):
        """
        Add an object annotation

        Args:
            image_id: ID of the image
            category_id: Category ID
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            segmentation_mask: Binary mask for the object
            area: Area of the object (computed from mask if not provided)
        """
        x_min, y_min, x_max, y_max = bbox

        # COCO format: [x, y, width, height]
        coco_bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        # Compute area
        if area is None:
            if segmentation_mask is not None:
                area = int(np.sum(segmentation_mask))
            else:
                area = coco_bbox[2] * coco_bbox[3]

        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": coco_bbox,
            "area": area,
            "iscrowd": 0,
        }

        # Add segmentation if mask is provided
        if segmentation_mask is not None:
            # Convert binary mask to polygon format using contour detection
            polygons = self._mask_to_polygons(segmentation_mask)
            if polygons:
                annotation["segmentation"] = polygons
            else:
                # Fallback to bbox if no valid polygons found
                annotation["segmentation"] = [
                    [
                        float(x_min),
                        float(y_min),
                        float(x_max),
                        float(y_min),
                        float(x_max),
                        float(y_max),
                        float(x_min),
                        float(y_max),
                    ]
                ]

        self.annotations["annotations"].append(annotation)
        self.annotation_id += 1

    def _mask_to_polygons(self, mask: np.ndarray, min_area: int = 10):
        """
        Convert binary mask to COCO polygon format
        
        Args:
            mask: Binary mask (HxW) where True/1 = object pixels
            min_area: Minimum contour area to keep
            
        Returns:
            List of polygons in COCO format [[x1,y1,x2,y2,...]]
        """
        try:
            import cv2
        except ImportError:
            # If opencv not available, return empty list (will use bbox fallback)
            return []
        
        # Ensure mask is uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < min_area:
                continue
            
            # Flatten contour to [x1, y1, x2, y2, ...]
            contour = contour.flatten().tolist()
            
            # COCO requires at least 6 values (3 points)
            if len(contour) >= 6:
                polygons.append(contour)
        
        return polygons

    def save(self, filename: str = "annotations.json"):
        """Save annotations to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.annotations, f, indent=2)

        print(f"COCO annotations saved to {output_path}")
        return output_path


class YOLOExporter:
    """Export annotations in YOLO format"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.labels_dir = self.output_dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = []

    def add_class(self, class_name: str) -> int:
        """
        Add a class name

        Returns:
            class_id (index in list)
        """
        if class_name not in self.class_names:
            self.class_names.append(class_name)
        return self.class_names.index(class_name)

    def add_annotation(
        self,
        image_name: str,
        image_width: int,
        image_height: int,
        class_id: int,
        bbox: Tuple[int, int, int, int],
    ):
        """
        Add an annotation for an image

        Args:
            image_name: Name of the image file (without extension)
            image_width: Width of the image
            image_height: Height of the image
            class_id: Class ID (index)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = bbox

        # Convert to YOLO format: [class_id, x_center, y_center, width, height]
        # All values normalized to 0-1
        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Clamp to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        # Write to label file (one line per object)
        # Ensure labels directory exists before writing
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        label_file = self.labels_dir / f"{image_name}.txt"
        with open(label_file, "a") as f:
            f.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )

    def save_class_names(self, filename: str = "classes.txt"):
        """Save class names to file"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

        print(f"YOLO class names saved to {output_path}")
        return output_path

    def create_yaml(
        self, train_path: str = None, val_path: str = None, filename: str = "data.yaml"
    ):
        """Create YAML configuration file for YOLO training"""
        yaml_content = f"""# YOLO dataset configuration
path: {str(self.output_dir)}
train: {train_path if train_path else 'images/train'}
val: {val_path if val_path else 'images/val'}

nc: {len(self.class_names)}
names: {self.class_names}
"""

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(yaml_content)

        print(f"YOLO config saved to {output_path}")
        return output_path


def compute_bbox_from_projection(
    vertices_3d: np.ndarray, mvp_matrix: np.ndarray, image_width: int, image_height: int
) -> Tuple[int, int, int, int] | None:
    """
    Compute 2D bounding box from 3D vertices projected to screen

    Args:
        vertices_3d: Nx3 array of 3D vertex positions
        mvp_matrix: Model-View-Projection matrix (4x4)
        image_width: Width of the output image
        image_height: Height of the output image

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates, or None if bbox is invalid
    """
    # Convert to homogeneous coordinates
    vertices_h = np.hstack([vertices_3d, np.ones((len(vertices_3d), 1))])

    # Project to clip space
    clip_coords = (mvp_matrix @ vertices_h.T).T

    # Perspective divide
    ndc_coords = clip_coords[:, :3] / clip_coords[:, 3:4]

    # Convert NDC (-1 to 1) to screen coordinates (0 to width/height)
    screen_x = (ndc_coords[:, 0] + 1) * 0.5 * image_width
    screen_y = (1 - ndc_coords[:, 1]) * 0.5 * image_height  # Flip Y

    # Compute bounding box
    x_min = max(0, int(np.min(screen_x)))
    x_max = min(image_width, int(np.max(screen_x)))
    y_min = max(0, int(np.min(screen_y)))
    y_max = min(image_height, int(np.max(screen_y)))

    # Validate bbox (must have positive area and be on screen)
    if (
        x_max <= x_min
        or y_max <= y_min
        or x_max <= 0
        or y_max <= 0
        or x_min >= image_width
        or y_min >= image_height
    ):
        return None

    return x_min, y_min, x_max, y_max
