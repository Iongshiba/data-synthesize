"""Engine configuration dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple
from .enums import (
    CameraMovement,
    ShapeType,
    ColorMode,
    ShadingModel,
    TextureMode,
    RenderMode,
    GradientMode,
    ModelVisualizationMode,
)


def _shader_path(*parts: str) -> str:
    return str(_SHADER_ROOT.joinpath(*parts).resolve())


_SHADER_ROOT = Path(__file__).resolve().parent.parent
_SHAPE_VERTEX_PATH = _shader_path("graphics", "phong.vert")
_SHAPE_FRAGMENT_PATH = _shader_path("graphics", "phong.frag")
_GOURAUD_VERTEX_PATH = _shader_path("graphics", "gouraud.vert")
_GOURAUD_FRAGMENT_PATH = _shader_path("graphics", "gouraud.frag")
_LIGHT_FRAGMENT_PATH = _shader_path("graphics", "light.frag")


# Model to texture mapping
MODEL_TEXTURE_MAP = {
    "catn0.obj": "cat_text_m.jpg",
    "Patchwork chair.ply": "Patchwork chair_0.jpg",
    "Christmas Bear.obj": "Christmas Bear_1.jpg",
    "DiamondSword.obj": "Diffuse.png",
}


@dataclass(slots=True)
class ShapeConfig:
    """Shape specific configuration attributes."""

    cylinder_height: float = 1.0
    cylinder_radius: float = 0.5
    cylinder_sectors: int = 3

    sphere_radius: float = 2.0
    sphere_sectors: int = 40
    sphere_stacks: int = 41
    # sphere_color: tuple[float, float, float] = (None, None, None)

    heart_sector: int = 64
    heart_stack: int = 32
    heart_scale: float = 1.0

    circle_sector: int = 100

    ellipse_sector: int = 100
    ellipse_a: int = 3
    ellipse_b: int = 1

    star_wing: int = 5
    star_outer_radius: int = 2
    star_inner_radius: int = 1

    cone_height: float = 1.0
    cone_radius: float = 0.5
    cone_sectors: int = 20

    truncated_height: float = 1.0
    truncated_top_radius: float = 0.3
    truncated_bottom_radius: float = 0.5
    truncated_sectors: int = 20

    torus_sectors: int = 50
    torus_stacks: int = 50
    torus_horizontal_radius: float = 2.0
    torus_vertical_radius: float = 1

    equation_expression: str = "(x^2 + y - 11)^2 + (x + y^2 - 7)^2"
    # equation_expression: str = "(1 - x)^2 + 100 * (y - x^2)^2"
    equation_mesh_size: int = 10
    equation_mesh_density: int = 100

    texture_file: str = "textures\wall.jpg"

    model_file: str = ""

    base_color: tuple[float | None, float | None, float | None] = (
        207,
        207,
        196,
    )

    # Gradient configuration
    gradient_mode: GradientMode = None
    gradient_start_color: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
    gradient_end_color: tuple[float, float, float] = (0.0, 0.0, 1.0)  # Blue


@dataclass(slots=True)
class CameraConfig:
    """Camera configuration parameters for initial setup."""

    position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    front: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    right: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    fov: float = 75.0
    near_plane: float = 0.1
    far_plane: float = 100.0
    move_speed: float = 0.25
    yaw: float = -90.0
    pitch: float = 0.0
    sensitivity: float = 0.05


@dataclass(slots=True)
class TrackballConfig:
    """Trackball configuration parameters for initial setup."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    distance: float = 10.0
    radians: bool = None
    pan_sensitivity: float = 0.001


@dataclass(slots=True)
class EngineConfig:
    """Aggregated configuration for the rendering engine."""

    width: int = 1000
    height: int = 1000
    camera: CameraConfig = field(default_factory=CameraConfig)
    trackball: TrackballConfig = field(default_factory=TrackballConfig)
    cull_face: bool = True
    # cull_face: bool = (
    #     False
    #     if shape
    #     in [
    #         ShapeType.TRIANGLE,
    #         ShapeType.RECTANGLE,
    #         ShapeType.PENTAGON,
    #         ShapeType.HEXAGON,
    #         ShapeType.CIRCLE,
    #         ShapeType.ELLIPSE,
    #         ShapeType.TRAPEZOID,
    #         ShapeType.STAR,
    #         ShapeType.ARROW,
    #         ShapeType.EQUATION,
    #     ]
    #     else True
    # )


__all__ = [
    "EngineConfig",
    "ShapeConfig",
    "CameraConfig",
    "TrackballConfig",
    "MODEL_TEXTURE_MAP",
    "_SHADER_ROOT",
    "_SHAPE_VERTEX_PATH",
    "_SHAPE_FRAGMENT_PATH",
    "_GOURAUD_VERTEX_PATH",
    "_GOURAUD_FRAGMENT_PATH",
    "_LIGHT_FRAGMENT_PATH",
    "CameraMovement",
    "ShapeType",
    "ColorMode",
    "ShadingModel",
    "TextureMode",
    "RenderMode",
    "GradientMode",
    "ModelVisualizationMode",
]
