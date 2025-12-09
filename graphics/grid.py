"""
Grid Plane - Blender-like infinite grid for navigation
"""

import numpy as np
from OpenGL import GL
from pathlib import Path
from graphics.shader import Shader, ShaderProgram


class GridPlane:
    """Infinite grid plane for visual navigation (not included in synthesis)"""

    def __init__(self, height: float = 0.0, scale: float = 1.0):
        self.height = height
        self.scale = scale
        self.enabled = True

        # Load shader program (vertex + fragment)
        shader_dir = Path(__file__).parent
        vert_path = str(shader_dir / "grid.vert")
        frag_path = str(shader_dir / "grid.frag")
        vert_shader = Shader(vert_path)
        frag_shader = Shader(frag_path)
        self.shader_program = ShaderProgram()
        self.shader_program.add_shader(vert_shader)
        self.shader_program.add_shader(frag_shader)
        self.shader_program.build()

        # Cache uniform locations
        self.proj_loc = GL.glGetUniformLocation(
            self.shader_program.program, "projection"
        )
        self.view_loc = GL.glGetUniformLocation(self.shader_program.program, "view")
        self.model_loc = GL.glGetUniformLocation(self.shader_program.program, "model")
        self.grid_height_loc = GL.glGetUniformLocation(
            self.shader_program.program, "gridHeight"
        )
        self.grid_scale_loc = GL.glGetUniformLocation(
            self.shader_program.program, "gridScale"
        )
        self.grid_thin_loc = GL.glGetUniformLocation(
            self.shader_program.program, "gridColorThin"
        )
        self.grid_thick_loc = GL.glGetUniformLocation(
            self.shader_program.program, "gridColorThick"
        )
        self.xaxis_loc = GL.glGetUniformLocation(
            self.shader_program.program, "xAxisColor"
        )
        self.zaxis_loc = GL.glGetUniformLocation(
            self.shader_program.program, "zAxisColor"
        )

        # Create a simple quad that covers the screen in NDC coordinates
        # The vertex shader will unproject these to create an infinite grid
        vertices = np.array(
            [
                # Positions (NDC)
                -1.0,
                -1.0,
                0.0,  # Bottom-left
                1.0,
                -1.0,
                0.0,  # Bottom-right
                1.0,
                1.0,
                0.0,  # Top-right
                -1.0,
                1.0,
                0.0,  # Top-left
            ],
            dtype=np.float32,
        )

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        # Create VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Vertex buffer
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )

        # Position attribute
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)

        # Element buffer
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW
        )

        GL.glBindVertexArray(0)

    def render(self, projection, view):
        """Render the grid"""
        if not self.enabled:
            return

        # Enable blending for transparency
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Activate shader program and set uniforms
        self.shader_program.activate()
        model = np.eye(4, dtype=np.float32)
        if self.proj_loc != -1:
            GL.glUniformMatrix4fv(self.proj_loc, 1, GL.GL_TRUE, projection)
        if self.view_loc != -1:
            GL.glUniformMatrix4fv(self.view_loc, 1, GL.GL_TRUE, view)
        if self.model_loc != -1:
            GL.glUniformMatrix4fv(self.model_loc, 1, GL.GL_TRUE, model)
        if self.grid_height_loc != -1:
            GL.glUniform1f(self.grid_height_loc, float(self.height))
        if self.grid_scale_loc != -1:
            GL.glUniform1f(self.grid_scale_loc, float(self.scale))
        # Colors
        if self.grid_thin_loc != -1:
            GL.glUniform4f(self.grid_thin_loc, 0.5, 0.5, 0.5, 0.5)
        if self.grid_thick_loc != -1:
            GL.glUniform4f(self.grid_thick_loc, 0.3, 0.3, 0.3, 1.0)
        if self.xaxis_loc != -1:
            GL.glUniform4f(self.xaxis_loc, 1.0, 0.0, 0.0, 1.0)
        if self.zaxis_loc != -1:
            GL.glUniform4f(self.zaxis_loc, 0.0, 0.0, 1.0, 1.0)

        # Draw
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

        self.shader_program.deactivate()
        GL.glDisable(GL.GL_BLEND)

    def cleanup(self):
        """Cleanup OpenGL resources"""
        try:
            if hasattr(self, "shader_program"):
                self.shader_program.cleanup()
        except Exception:
            pass
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteBuffers(1, [self.ebo])
