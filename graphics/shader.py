from OpenGL import GL


class Shader:
    def __init__(self, source, typ=None):
        self.source = source

        if typ is None:
            typ_name = source.split(".")[-1]
            match typ_name:
                case "vert":
                    typ = GL.GL_VERTEX_SHADER
                case "frag":
                    typ = GL.GL_FRAGMENT_SHADER
                case _:
                    raise ValueError(f"Invalid file extension {typ_name}")

        self.typ = typ

        self.shader = self.compile_shader(source, typ)

    @staticmethod
    def compile_shader(source, typ):
        with open(source, "r") as f:
            code = f.read()
            shader = GL.glCreateShader(typ)
            GL.glShaderSource(shader, code)
            GL.glCompileShader(shader)

        if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            log = GL.glGetShaderInfoLog(shader).decode()
            raise RuntimeError(log)

        return shader


class ShaderProgram:
    def __init__(self):
        self.shaders = {}
        self.program = GL.glCreateProgram()

    def add_shader(self, shader):
        self.shaders[shader.source] = shader.shader

    def build(self):
        for _, shader in self.shaders.items():
            GL.glAttachShader(self.program, shader)
        GL.glLinkProgram(self.program)
        for _, shader in self.shaders.items():
            GL.glDeleteShader(shader)

        if GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            log = GL.glGetProgramInfoLog(self.program).decode()
            raise RuntimeError(log)

    def activate(self):
        GL.glUseProgram(self.program)

    def deactivate(self):
        GL.glUseProgram(0)

    def cleanup(self):
        """Delete the shader program to free GPU resources."""
        try:
            if self.program is not None:
                GL.glDeleteProgram(self.program)
                self.program = None
        except (GL.error.GLError, AttributeError, TypeError):
            pass

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()
