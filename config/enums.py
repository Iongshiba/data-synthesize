"""Core enumerations for engine configuration and rendering options."""

from __future__ import annotations

from enum import Enum, auto


class CameraMovement(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


class ShapeType(Enum):
    # fmt: off
    TRIANGLE = auto()
    RECTANGLE = auto()
    PENTAGON = auto()           
    HEXAGON = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    TRAPEZOID = auto()
    STAR = auto()
    ARROW = auto()
    CUBE = auto()
    SPHERE = auto()
    HEART = auto()
    CYLINDER = auto()
    CONE = auto()              
    TRUNCATED_CONE = auto()    
    TETRAHEDRON = auto()
    TORUS = auto()
    QUICK_DRAW = auto()     
    EQUATION = auto()
    MODEL = auto()
    LIGHT_SOURCE = auto()


class ColorMode(Enum):
    FLAT = 0
    VERTEX = 1


class ShadingModel(Enum):
    NORMAL = 0
    PHONG = 1
    GOURAUD = 2


class TextureMode(Enum):
    NONE = 0
    ENABLED = 1


class RenderMode(Enum):
    FILL = 0
    WIREFRAME = 1


class GradientMode(Enum):
    NONE = 0
    LINEAR_X = 1  # Gradient along X axis
    LINEAR_Y = 2  # Gradient along Y axis
    LINEAR_Z = 3  # Gradient along Z axis
    RADIAL = 4  # Gradient from center
    DIAGONAL = 5  # Gradient along diagonal
    RAINBOW = 6  # Rainbow gradient


class ModelVisualizationMode(Enum):
    NORMAL = 0  # Normal rendering
    BOUNDING_BOX = 1  # Display bounding box
    DEPTH_MAP = 2  # Display depth map
    SEGMENTATION_MASK = 3  # Display segmentation mask
