"""
Data Synthesis Scene - Main scene for generating training data
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import uuid

from graphics.scene import Node, TransformNode, GeometryNode
from shape.mesh_loader import MeshShape
from rendering.world import Translate, Scale, Rotate, Composite


class SynthesisObject:
    """Represents an object in the synthesis scene"""

    def __init__(
        self,
        name: str,
        shape: MeshShape,
        position: np.ndarray = None,
        rotation: np.ndarray = None,
        scale: float = 1.0,
        class_id: int = 0,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.shape = shape
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0])
        self.rotation = (
            rotation if rotation is not None else np.array([0.0, 0.0, 0.0])
        )  # Euler angles (x, y, z)
        self.scale_factor = scale
        self.class_id = class_id
        self.visible = True

        # Unique color for segmentation (based on object ID)
        self.segmentation_color = self._generate_segmentation_color()

    def _generate_segmentation_color(self) -> np.ndarray:
        """Generate a unique color for segmentation based on object ID"""
        # Use hash of ID to generate consistent color
        hash_val = hash(self.id)
        r = ((hash_val & 0xFF0000) >> 16) / 255.0
        g = ((hash_val & 0x00FF00) >> 8) / 255.0
        b = (hash_val & 0x0000FF) / 255.0
        return np.array([r, g, b])

    def get_transform(self):
        """Get the composite transform for this object"""
        transforms = []

        # Apply in order: Scale -> Rotate -> Translate
        if self.scale_factor != 1.0:
            transforms.append(
                Scale(self.scale_factor, self.scale_factor, self.scale_factor)
            )

        # Apply rotations (X, Y, Z order)
        if not np.allclose(self.rotation, 0.0):
            if self.rotation[0] != 0:
                transforms.append(Rotate(axis=(1, 0, 0), angle=self.rotation[0]))
            if self.rotation[1] != 0:
                transforms.append(Rotate(axis=(0, 1, 0), angle=self.rotation[1]))
            if self.rotation[2] != 0:
                transforms.append(Rotate(axis=(0, 0, 1), angle=self.rotation[2]))

        # Translation
        if not np.allclose(self.position, 0.0):
            transforms.append(
                Translate(self.position[0], self.position[1], self.position[2])
            )

        if len(transforms) == 0:
            from rendering.world import Transform

            return Transform()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            return Composite(transforms)

    def get_world_bounding_box(self) -> Dict:
        """Get bounding box in world coordinates"""
        bbox = self.shape.get_bounding_box()

        # Apply transformations to bbox
        min_transformed = bbox["min"] * self.scale_factor + self.position
        max_transformed = bbox["max"] * self.scale_factor + self.position

        return {
            "min": min_transformed,
            "max": max_transformed,
            "center": (min_transformed + max_transformed) / 2,
            "size": max_transformed - min_transformed,
        }


class SynthesisObjectGroup:
    """Group of related objects (e.g., from a single FBX file with multiple meshes)"""

    def __init__(self, name: str, class_id: int = 0):
        self.id = str(uuid.uuid4())
        self.name = name
        self.class_id = class_id
        self.objects: List[SynthesisObject] = []
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.scale_factor = 1.0
        self.visible = True

    def add_object(self, obj: SynthesisObject):
        """Add an object to this group"""
        self.objects.append(obj)

    def get_transform(self):
        """Get the composite transform for the entire group"""
        transforms = []

        if self.scale_factor != 1.0:
            transforms.append(
                Scale(self.scale_factor, self.scale_factor, self.scale_factor)
            )

        if not np.allclose(self.rotation, 0.0):
            if self.rotation[0] != 0:
                transforms.append(Rotate(axis=(1, 0, 0), angle=self.rotation[0]))
            if self.rotation[1] != 0:
                transforms.append(Rotate(axis=(0, 1, 0), angle=self.rotation[1]))
            if self.rotation[2] != 0:
                transforms.append(Rotate(axis=(0, 0, 1), angle=self.rotation[2]))

        if not np.allclose(self.position, 0.0):
            transforms.append(
                Translate(self.position[0], self.position[1], self.position[2])
            )

        if len(transforms) == 0:
            from rendering.world import Transform

            return Transform()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            return Composite(transforms)


class SynthesisScene:
    """Scene manager for data synthesis"""

    def __init__(self):
        self.objects: List[SynthesisObject] = []
        self.object_groups: List[SynthesisObjectGroup] = []
        self.class_names: Dict[int, str] = {}
        self.next_class_id = 0

        self.root = Node("synthesis_root")

    def add_object(
        self,
        mesh_path: str,
        class_name: str,
        position: np.ndarray = None,
        rotation: np.ndarray = None,
        scale: float = 1.0,
    ) -> SynthesisObject:
        """Add a candidate object to the scene"""
        # Get or create class ID
        class_id = self._get_class_id(class_name)

        # Create the shape
        shape = MeshShape(mesh_path, scale=1.0, color=(0.8, 0.2, 0.2))

        # Create synthesis object
        obj = SynthesisObject(
            name=f"{class_name}_{len(self.objects)}",
            shape=shape,
            position=position if position is not None else np.array([0.0, 0.0, 0.0]),
            rotation=rotation if rotation is not None else np.array([0.0, 0.0, 0.0]),
            scale=scale,
            class_id=class_id,
        )

        self.objects.append(obj)
        self._rebuild_scene()

        return obj

    def add_object_group(
        self, group_name: str, class_name: str
    ) -> SynthesisObjectGroup:
        """Create a new object group"""
        class_id = self._get_class_id(class_name)
        group = SynthesisObjectGroup(group_name, class_id)
        self.object_groups.append(group)
        return group

    def remove_object(self, obj: SynthesisObject) -> None:
        """Remove an object from the scene"""
        if obj in self.objects:
            self.objects.remove(obj)
            self._rebuild_scene()

    def clear_objects(self) -> None:
        """Remove all objects and groups"""
        self.objects.clear()
        self.object_groups.clear()
        self._rebuild_scene()

    def _get_class_id(self, class_name: str) -> int:
        """Get or create a class ID for a class name"""
        for class_id, name in self.class_names.items():
            if name == class_name:
                return class_id

        # Create new class ID
        class_id = self.next_class_id
        self.class_names[class_id] = class_name
        self.next_class_id += 1
        return class_id

    def _rebuild_scene(self) -> None:
        """Rebuild the scene graph from current state"""
        # Clear children
        self.root.children.clear()

        # Add standalone objects
        for obj in self.objects:
            if obj.visible:
                transform_node = TransformNode(
                    name=f"transform_{obj.name}",
                    transform=obj.get_transform(),
                    children=[GeometryNode(obj.name, obj.shape)],
                )
                self.root.add(transform_node)

        # Add object groups with hierarchical transforms
        for group in self.object_groups:
            if group.visible:
                # Group transform node
                group_children = []
                for obj in group.objects:
                    if obj.visible:
                        # Object's own transform relative to group
                        obj_transform_node = TransformNode(
                            name=f"transform_{obj.name}",
                            transform=obj.get_transform(),
                            children=[GeometryNode(obj.name, obj.shape)],
                        )
                        group_children.append(obj_transform_node)

                # Wrap all objects in group transform
                group_transform_node = TransformNode(
                    name=f"transform_group_{group.name}",
                    transform=group.get_transform(),
                    children=group_children,
                )
                self.root.add(group_transform_node)

    def get_root(self) -> Node:
        """Get the root node of the scene"""
        return self.root

    def get_object_list(self) -> List[Dict]:
        """Get list of objects for UI display"""
        return [
            {
                "id": obj.id,
                "name": obj.name,
                "class": self.class_names.get(obj.class_id, "unknown"),
                "position": obj.position.tolist(),
                "visible": obj.visible,
            }
            for obj in self.objects
        ]
