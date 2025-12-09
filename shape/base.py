import numpy as np

from pathlib import Path
from OpenGL import GL

from utils import *
from config import (
    _SHAPE_FRAGMENT_PATH,
    _SHAPE_VERTEX_PATH,
    _GOURAUD_VERTEX_PATH,
    _GOURAUD_FRAGMENT_PATH,
    ShadingModel,
)
from graphics.buffer import VAO
from graphics.shader import Shader, ShaderProgram
from graphics.texture import Texture2D


class Part:
    def __init__(
        self,
        vao: VAO,
        draw_mode: GL.constant.IntConstant,
        vertex_num: int,
        index_num: int | None = None,
    ):
        self.vao = vao
        self.draw_mode = draw_mode
        self.vertex_num = vertex_num
        self.index_num = index_num


# fmt: on
class Shape:
    def __init__(self, vertex_file: str, fragment_file: str):
        # Shaders
        if not vertex_file:
            vertex_file = _SHAPE_VERTEX_PATH
        if not fragment_file:
            fragment_file = _SHAPE_FRAGMENT_PATH

        vertex_shader = Shader(vertex_file)
        fragment_shader = Shader(fragment_file)
        self.shader_program = ShaderProgram()
        self.shader_program.add_shader(vertex_shader)
        self.shader_program.add_shader(fragment_shader)
        self.shader_program.build()

        # Geometry containers
        self.shapes: list[Part] = []

        self.identity = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.texture = None
        self.texture_enabled = False
        self.shading_mode = ShadingModel.PHONG

        # fmt: off
        self.transform_loc = GL.glGetUniformLocation(self.shader_program.program, "transform")
        self.camera_loc = GL.glGetUniformLocation(self.shader_program.program, "camera")
        self.project_loc = GL.glGetUniformLocation(self.shader_program.program, "project")
        self.use_texture_loc = GL.glGetUniformLocation(self.shader_program.program, "use_texture")
        self.texture_data_loc = GL.glGetUniformLocation(self.shader_program.program, "textureData")
        # Phong (eye-space) uniforms
        self.I_lights_loc = GL.glGetUniformLocation(self.shader_program.program, "I_lights")
        self.K_materials_loc = GL.glGetUniformLocation(self.shader_program.program, "K_materials")
        self.shininess_loc = GL.glGetUniformLocation(self.shader_program.program, "shininess")
        self.light_coord_loc = GL.glGetUniformLocation(self.shader_program.program, "lightCoord")
        self.shading_mode_loc = GL.glGetUniformLocation(self.shader_program.program, "shadingMode")

        self.shader_program.activate()
        GL.glUniformMatrix4fv(self.transform_loc, 1, GL.GL_TRUE, self.identity)
        GL.glUniformMatrix4fv(self.camera_loc, 1, GL.GL_TRUE, self.identity)
        GL.glUniformMatrix4fv(self.project_loc, 1, GL.GL_TRUE, self.identity)
        GL.glUniform1i(self.use_texture_loc, True)
        GL.glUniform1i(self.texture_data_loc, 0)
        if self.shading_mode_loc != -1:
            GL.glUniform1i(self.shading_mode_loc, self.shading_mode.value)

        # Set sensible defaults for Phong-eye uniforms if present
        # Default shininess
        if self.shininess_loc != -1:
            GL.glUniform1f(self.shininess_loc, 32.0)

        # Default light/material matrices: columns are [diffuse, specular, unused]
        if self.I_lights_loc != -1:
            I = np.zeros((3, 3), dtype=np.float32)
            I[:, 0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # diffuse intensity (RGB)
            I[:, 1] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # specular intensity (RGB)
            I[:, 2] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            GL.glUniformMatrix3fv(self.I_lights_loc, 1, GL.GL_TRUE, I)

        if self.K_materials_loc != -1:
            K = np.zeros((3, 3), dtype=np.float32)
            K[:, 0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Kd (diffuse reflectance)
            K[:, 1] = np.array([0.3, 0.3, 0.3], dtype=np.float32)  # Ks (specular reflectance)
            K[:, 2] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            GL.glUniformMatrix3fv(self.K_materials_loc, 1, GL.GL_TRUE, K)
        self.shader_program.deactivate()

    def draw(self):
        self.shader_program.activate()
        # Set use_texture based on whether texture exists and is enabled
        GL.glUniform1i(
            self.use_texture_loc, 1 if (self.texture and self.texture_enabled) else 0
        )
        for shape in self.shapes:
            vao = shape.vao
            vao.activate()
            if self.texture and self.texture_enabled:
                self.texture.activate()
            # fmt: off
            if vao.ebo is not None:
                GL.glDrawElements(
                    shape.draw_mode, shape.index_num, GL.GL_UNSIGNED_INT, None
                )
            else:
                GL.glDrawArrays(
                    shape.draw_mode, 0, shape.vertex_num
                )
            if self.texture and self.texture_enabled:
                self.texture.deactivate()
            vao.deactivate()
        self.shader_program.deactivate()

    def transform(
        self,
        project_matrix: np.ndarray,
        view_matrix: np.ndarray,
        model_matrix: np.ndarray,
    ):
        self.shader_program.activate()
        GL.glUniformMatrix4fv(self.project_loc, 1, GL.GL_TRUE, project_matrix)
        GL.glUniformMatrix4fv(self.camera_loc, 1, GL.GL_TRUE, view_matrix)
        GL.glUniformMatrix4fv(self.transform_loc, 1, GL.GL_TRUE, model_matrix)
        self.shader_program.deactivate()

    def lighting(
        self,
        light_color: np.ndarray,
        light_position: np.ndarray,
        camera_position: np.ndarray,
    ):
        self.shader_program.activate()

        # Columns correspond to [diffuse, specular, unused].
        if self.I_lights_loc != -1:
            I = np.zeros((3, 3), dtype=np.float32)
            I[:, 0] = np.array(light_color, dtype=np.float32)
            I[:, 1] = np.array(light_color, dtype=np.float32)
            I[:, 2] = np.array(light_color, dtype=np.float32)
            GL.glUniformMatrix3fv(self.I_lights_loc, 1, GL.GL_TRUE, I)

        # material coefficients modify this method in the specific shape class.
        if self.K_materials_loc != -1:
            K = np.zeros((3, 3), dtype=np.float32)
            # diffuse
            K[:, 0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            # specular
            K[:, 1] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            # ambient
            K[:, 2] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            GL.glUniformMatrix3fv(self.K_materials_loc, 1, GL.GL_TRUE, K)

        if self.shininess_loc != -1:
            GL.glUniform1f(self.shininess_loc, 32.0)

        # light position should be provided in eye-space
        if self.light_coord_loc != -1:
            GL.glUniform3fv(self.light_coord_loc, 1, light_position)
        self.shader_program.deactivate()

    def set_shading_mode(self, shading: ShadingModel) -> None:
        if shading == self.shading_mode:
            return

        # Determine shader paths based on shading mode
        if shading == ShadingModel.GOURAUD:
            vertex_path = _GOURAUD_VERTEX_PATH
            fragment_path = _GOURAUD_FRAGMENT_PATH
        elif shading == ShadingModel.PHONG:
            vertex_path = _SHAPE_VERTEX_PATH
            fragment_path = _SHAPE_FRAGMENT_PATH
        else:  # NORMAL
            vertex_path = _SHAPE_VERTEX_PATH
            fragment_path = _SHAPE_FRAGMENT_PATH

        self.shading_mode = shading
        self._reload_shaders(vertex_path, fragment_path)

    def _reload_shaders(self, vertex_file: str, fragment_file: str):
        """Reload shaders with new vertex and fragment shader files."""
        # Clean up old shader program
        if hasattr(self.shader_program, "cleanup"):
            self.shader_program.cleanup()

        # Create new shader program
        vertex_shader = Shader(vertex_file)
        fragment_shader = Shader(fragment_file)
        self.shader_program = ShaderProgram()
        self.shader_program.add_shader(vertex_shader)
        self.shader_program.add_shader(fragment_shader)
        self.shader_program.build()

        # Re-query all uniform locations
        self.transform_loc = GL.glGetUniformLocation(
            self.shader_program.program, "transform"
        )
        self.camera_loc = GL.glGetUniformLocation(self.shader_program.program, "camera")
        self.project_loc = GL.glGetUniformLocation(
            self.shader_program.program, "project"
        )
        self.use_texture_loc = GL.glGetUniformLocation(
            self.shader_program.program, "use_texture"
        )
        self.texture_data_loc = GL.glGetUniformLocation(
            self.shader_program.program, "textureData"
        )
        self.I_lights_loc = GL.glGetUniformLocation(
            self.shader_program.program, "I_lights"
        )
        self.K_materials_loc = GL.glGetUniformLocation(
            self.shader_program.program, "K_materials"
        )
        self.shininess_loc = GL.glGetUniformLocation(
            self.shader_program.program, "shininess"
        )
        self.light_coord_loc = GL.glGetUniformLocation(
            self.shader_program.program, "lightCoord"
        )
        self.shading_mode_loc = GL.glGetUniformLocation(
            self.shader_program.program, "shadingMode"
        )

        # Re-initialize uniforms with defaults
        self.shader_program.activate()
        GL.glUniformMatrix4fv(self.transform_loc, 1, GL.GL_TRUE, self.identity)
        GL.glUniformMatrix4fv(self.camera_loc, 1, GL.GL_TRUE, self.identity)
        GL.glUniformMatrix4fv(self.project_loc, 1, GL.GL_TRUE, self.identity)
        GL.glUniform1i(self.use_texture_loc, True)
        GL.glUniform1i(self.texture_data_loc, 0)

        if self.shading_mode_loc != -1:
            GL.glUniform1i(self.shading_mode_loc, self.shading_mode.value)

        if self.shininess_loc != -1:
            GL.glUniform1f(self.shininess_loc, 32.0)

        if self.I_lights_loc != -1:
            I = np.zeros((3, 3), dtype=np.float32)
            I[:, 0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            I[:, 1] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            I[:, 2] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            GL.glUniformMatrix3fv(self.I_lights_loc, 1, GL.GL_TRUE, I)

        if self.K_materials_loc != -1:
            K = np.zeros((3, 3), dtype=np.float32)
            K[:, 0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            K[:, 1] = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            K[:, 2] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            GL.glUniformMatrix3fv(self.K_materials_loc, 1, GL.GL_TRUE, K)

        self.shader_program.deactivate()

    @staticmethod
    def _apply_color_override(
        colors: np.ndarray,
        override: tuple[float | None, float | None, float | None] | None,
    ) -> np.ndarray:
        if not override:
            return colors

        for idx, channel in enumerate(override):
            if channel is not None:
                colors[:, idx] = channel
        return colors

    def _create_texture(self, path):
        img_data, width, height = load_texture(path)
        self.texture = Texture2D()
        self.texture.add_texture(
            img_data,
            width,
            height,
        )
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def set_texture_enabled(self, enabled: bool) -> None:
        """Enable or disable texture mapping for this shape."""
        self.texture_enabled = enabled

    def cleanup(self):
        """Cleanup OpenGL resources used by this shape."""
        try:
            # Clean up all VAOs
            for part in self.shapes:
                if hasattr(part.vao, "cleanup"):
                    part.vao.cleanup()

            # Clean up texture if exists
            if self.texture and hasattr(self.texture, "cleanup"):
                self.texture.cleanup()

            # Clean up shader program
            if hasattr(self.shader_program, "cleanup"):
                self.shader_program.cleanup()
        except Exception:
            pass  # Silently ignore cleanup errors
