from __future__ import annotations

from OpenGL import GL

from config import ShadingModel
from graphics.scene import Node, LightNode, GeometryNode, TransformNode
from rendering.camera import Camera, CameraMovement, Trackball
from rendering.world import Transform


class Renderer:
    def __init__(self, config):
        self.config = config
        self.camera = Camera(config.camera)
        self.trackball = Trackball(config.trackball)

        self.app = None
        self.root = None

        # GL state (simple defaults)
        GL.glViewport(0, 0, self.config.width, self.config.height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        if config.cull_face:
            GL.glEnable(GL.GL_CULL_FACE)
        # Use back-face culling by default so correctly wound meshes are visible
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        GL.glClearColor(0.2, 0.2, 0.2, 1.0)

        self.use_trackball = False
        self.use_wireframe = False
        self.use_texture = True
        self.shading_model = ShadingModel.PHONG
        self.cull_face_enabled = config.cull_face

        self.shape_nodes = []
        self.light_nodes = []
        self.transform_nodes = []

    def set_scene(self, scene):
        self.root = scene

    def _collect_node(self, node):
        if isinstance(node, LightNode):
            self.light_nodes.append(node)
        elif isinstance(node, GeometryNode):
            self.shape_nodes.append(node)
        elif isinstance(node, TransformNode):
            self.transform_nodes.append(node)
        for child in node.children:
            self._collect_node(child)

    def _apply_lighting(self):
        if not self.light_nodes:
            return

        if self.shading_model is ShadingModel.NORMAL:
            return

        light = self.light_nodes[0].shape

        # Not use anymore, should be removed
        camera_position = (
            self.trackball.get_camera_position()
            if self.use_trackball
            else self.camera.position
        )

        for node in self.shape_nodes:
            node.shape.lighting(
                light.get_color(),
                light.get_position(),
                camera_position,
            )

    def _apply_shading(self):
        for node in self.shape_nodes:
            if hasattr(node.shape, "set_shading_mode"):
                node.shape.set_shading_mode(self.shading_model)

    def _apply_animation(self, dt):
        for node in self.transform_nodes:
            node.transform.update_matrix(dt)

    def render(self, delta_time):
        if not self.app:
            raise ValueError("Must attach to an Application")

        if self.root is None:
            return

        aspect_ratio = (
            float(self.app.get_aspect_ratio())
            if self.app and hasattr(self.app, "get_aspect_ratio")
            else float(self.config.width) / float(self.config.height)
        )
        self.camera.aspect_ratio = aspect_ratio

        projection_matrix = (
            self.camera.get_projection_matrix()
            if not self.use_trackball
            else self.trackball.get_projection_matrix(self.app.winsize)
        )
        view_matrix = (
            self.camera.get_view_matrix()
            if not self.use_trackball
            else self.trackball.get_view_matrix()
        )

        if self.app:
            width, height = self.app.winsize
            GL.glViewport(0, 0, int(width), int(height))

        self.shape_nodes.clear()
        self.light_nodes.clear()
        self.transform_nodes.clear()
        self._collect_node(self.root)
        self._apply_shading()
        self._apply_animation(delta_time)
        self._apply_lighting()
        self.root.draw(None, view_matrix, projection_matrix)

        # Render grid plane if available (after scene, for transparency)
        if self.app and hasattr(self.app, "grid_plane") and self.app.grid_plane.enabled:
            self.app.grid_plane.render(projection_matrix, view_matrix)

    def move_camera(self, movement: CameraMovement, step_scale: float = 1.0) -> None:
        self.camera.move(movement, step_scale)

    def rotate_camera(self, old, new):
        self.camera.look(old, new)

    def pan_camera(self, old, new):
        self.camera.pan(old, new)

    def move_trackball(self, old, new):
        self.trackball.pan(old, new)

    def rotate_trackball(self, old, new, winsize):
        self.trackball.drag(old, new, winsize)

    def zoom_trackball(self, delta, winsize):
        self.trackball.zoom(delta, winsize)

    def toggle_wireframe(self):
        self.use_wireframe = False if self.use_wireframe else True
        if self.use_wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    def toggle_texture_mapping(self):
        self.use_texture = not self.use_texture
        # Apply to all shapes that have textures
        for node in self.shape_nodes:
            if hasattr(node.shape, "set_texture_enabled"):
                node.shape.set_texture_enabled(self.use_texture)

    def set_shading_model(self, shading: ShadingModel) -> None:
        self.shading_model = shading

    def set_face_culling(self, enabled: bool) -> None:
        if enabled and not self.cull_face_enabled:
            GL.glEnable(GL.GL_CULL_FACE)
        elif not enabled and self.cull_face_enabled:
            GL.glDisable(GL.GL_CULL_FACE)
        self.cull_face_enabled = enabled

    def cleanup(self):
        """Cleanup all OpenGL resources."""
        try:
            # Cleanup all shapes in the scene
            if self.root:
                self._cleanup_node(self.root)

            # Clear node lists
            self.shape_nodes.clear()
            self.light_nodes.clear()
            self.transform_nodes.clear()
            self.root = None
        except Exception:
            pass  # Silently ignore cleanup errors

    def _cleanup_node(self, node):
        """Recursively cleanup all nodes in the scene tree."""
        try:
            if hasattr(node, "shape") and node.shape and hasattr(node.shape, "cleanup"):
                node.shape.cleanup()

            for child in node.children:
                self._cleanup_node(child)
        except Exception:
            pass
