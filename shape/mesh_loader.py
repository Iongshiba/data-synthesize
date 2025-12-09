"""
Mesh loader for .obj and .ply files to create Shape objects
"""

import numpy as np
from pathlib import Path
from OpenGL import GL

from shape.base import Shape, Part
from graphics.buffer import VAO
from graphics.vertex import Vertex
from utils.misc import load_model


class MeshShape(Shape):
    """Shape created from loading a mesh file (.obj, .ply)"""

    def __init__(
        self,
        mesh_path: str,
        vertex_file: str = None,
        fragment_file: str = None,
        color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        scale: float = 1.0,
    ):
        """
        Load a mesh from file and create a Shape

        Args:
            mesh_path: Path to the .obj or .ply file
            vertex_file: Custom vertex shader (optional)
            fragment_file: Custom fragment shader (optional)
            color: RGB color for the mesh (0-1 range)
            scale: Scale factor for the mesh
        """
        super().__init__(vertex_file, fragment_file)

        self.mesh_path = Path(mesh_path)
        self.base_color = np.array(color, dtype=np.float32)
        self.original_color = np.array(
            color, dtype=np.float32
        )  # Store for unhighlighting
        self.scale = scale
        self.materials = {}
        self.textures = {}  # Store loaded textures
        self.mesh_vertices = []  # Store all vertices for projection
        self.is_highlighted = False

        # Load the mesh data
        meshes = load_model(str(mesh_path))

        # Store bounding box info
        self._compute_bounding_box(meshes)

        # Extract materials if available
        if meshes and "materials" in meshes[0]:
            self.materials = meshes[0].get("materials", {})

        # Create OpenGL objects from mesh data
        for mesh_data in meshes:
            vertices = mesh_data["vertices"] * scale
            normals = mesh_data["normals"]
            tex_coords = mesh_data["tex_coords"]
            indices = mesh_data["indices"]

            # Store vertices for later projection
            self.mesh_vertices.append(vertices)

            # Create per-vertex colors (use base_color for all vertices)
            num_vertices = len(vertices)
            colors = np.tile(self.base_color, (num_vertices, 1))

            # Create VAO
            vao = VAO()
            vao.add_vbo(
                0, vertices.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None
            )  # position
            vao.add_vbo(
                1, colors.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None
            )  # color
            vao.add_vbo(
                2, normals.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None
            )  # normal
            vao.add_vbo(
                3, tex_coords.flatten(), 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None
            )  # texture coords

            if len(indices) > 0:
                vao.add_ebo(indices)
                part = Part(vao, GL.GL_TRIANGLES, num_vertices, len(indices))
            else:
                part = Part(vao, GL.GL_TRIANGLES, num_vertices)

            self.shapes.append(part)

    def _compute_bounding_box(self, meshes):
        """Compute axis-aligned bounding box for all meshes"""
        all_vertices = []
        for mesh_data in meshes:
            all_vertices.append(mesh_data["vertices"])

        if all_vertices:
            all_verts = np.vstack(all_vertices)
            self.bbox_min = np.min(all_verts, axis=0) * self.scale
            self.bbox_max = np.max(all_verts, axis=0) * self.scale
            self.bbox_center = (self.bbox_min + self.bbox_max) / 2
            self.bbox_size = self.bbox_max - self.bbox_min
        else:
            self.bbox_min = np.zeros(3)
            self.bbox_max = np.zeros(3)
            self.bbox_center = np.zeros(3)
            self.bbox_size = np.zeros(3)

    def get_bounding_box(self):
        """Return bounding box information"""
        return {
            "min": self.bbox_min,
            "max": self.bbox_max,
            "center": self.bbox_center,
            "size": self.bbox_size,
        }

    def get_all_vertices(self):
        """Return all vertices from all mesh parts"""
        if not self.mesh_vertices:
            return np.array([])
        return np.vstack(self.mesh_vertices)

    def set_color(self, color: tuple[float, float, float]):
        """Change the color of the mesh"""
        self.base_color = np.array(color, dtype=np.float32)

        # Update colors in all VAOs
        for part in self.shapes:
            num_vertices = part.vertex_num
            colors = np.tile(self.base_color, (num_vertices, 1))

            # Update VBO at location 1 (color)
            part.vao.add_vbo(1, colors.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

    def set_highlight(self, highlighted: bool):
        """Highlight or unhighlight the object by brightening its color"""
        # Be defensive: some older instances may not have is_highlighted set
        currently = getattr(self, "is_highlighted", False)
        if highlighted and not currently:
            # Brighten the color (increase by 40%)
            base = getattr(self, "original_color", self.base_color)
            highlight_color = np.minimum(base * 1.4, 1.0)
            self.set_color(tuple(highlight_color))
            self.is_highlighted = True
        elif not highlighted and currently:
            # Restore original color
            base = getattr(self, "original_color", self.base_color)
            self.set_color(tuple(base))
            self.is_highlighted = False

    def load_texture(self, texture_path: str, material_name: str = None):
        """
        Load a texture from file and bind it to the mesh

        Args:
            texture_path: Path to the texture file (.png, .jpg, etc.)
            material_name: Optional material name to associate with this texture
        """
        from utils.misc import load_texture

        try:
            img_data, width, height = load_texture(texture_path)

            # Generate OpenGL texture
            texture_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

            # Set texture parameters
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            # Upload texture data
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA,
                width,
                height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                img_data,
            )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

            # Store texture ID
            key = material_name if material_name else "default"
            self.textures[key] = texture_id

            print(f"Loaded texture: {texture_path} ({width}x{height})")
            return texture_id

        except Exception as e:
            print(f"Failed to load texture {texture_path}: {e}")
            return None

    def apply_material_textures(self):
        """Load all textures from the material library"""
        if not self.materials:
            print("No materials found")
            return
        base_path = self.mesh_path.parent

        # Candidate directories to search for textures (mesh dir, meshdir/textures, parent/textures)
        candidate_dirs = [
            base_path,
            base_path / "textures",
            base_path.parent / "textures",
            base_path.parent,
        ]

        for mat_name, mat_data in self.materials.items():
            diffuse_ref = mat_data.get("diffuse_texture")

            if diffuse_ref:
                # Try direct resolution first
                resolved = None
                for d in candidate_dirs:
                    try_path = d / diffuse_ref
                    if try_path.exists():
                        resolved = try_path
                        break

                # If not found, try case-insensitive match by filename only
                if resolved is None:
                    for d in candidate_dirs:
                        if not d.exists():
                            continue
                        for f in d.iterdir():
                            if (
                                f.is_file()
                                and f.name.lower() == Path(diffuse_ref).name.lower()
                            ):
                                resolved = f
                                break
                        if resolved:
                            break

                if resolved:
                    self.load_texture(str(resolved), mat_name)
                else:
                    print(f"Texture not found for material '{mat_name}': {diffuse_ref}")
            else:
                # If no diffuse texture referenced, attempt to auto-load any textures in candidate dirs
                for d in candidate_dirs:
                    if not d.exists():
                        continue
                    for f in (
                        sorted(d.glob("*.png"))
                        + sorted(d.glob("*.jpg"))
                        + sorted(d.glob("*.jpeg"))
                    ):
                        # Use texture file stem as material key when material unknown
                        key = mat_name if mat_name else f.stem
                        if key not in self.textures:
                            self.load_texture(str(f), key)

    def draw(self, force_no_texture=False):
        """Override draw to bind textures from self.textures dict.

        Args:
            force_no_texture: If True, disable texture mapping even if textures are loaded
        """
        self.shader_program.activate()

        # Check if we have any textures loaded
        has_texture = len(self.textures) > 0 and not force_no_texture

        # Set use_texture uniform
        GL.glUniform1i(self.use_texture_loc, 1 if has_texture else 0)

        # If we have textures, bind the first one to texture unit 0
        if has_texture:
            # Get first texture from dict
            texture_id = next(iter(self.textures.values()))
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            GL.glUniform1i(self.texture_data_loc, 0)  # Use texture unit 0

        # Draw all parts
        for shape in self.shapes:
            vao = shape.vao
            vao.activate()

            if vao.ebo is not None:
                GL.glDrawElements(
                    shape.draw_mode, shape.index_num, GL.GL_UNSIGNED_INT, None
                )
            else:
                GL.glDrawArrays(shape.draw_mode, 0, shape.vertex_num)

            vao.deactivate()

        # Unbind texture if we used one
        if has_texture:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.shader_program.deactivate()
