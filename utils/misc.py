# import pyassimp
import sympy as sp
import numpy as np
import pyassimp
from OpenGL import GL
from PIL import Image
from plyfile import PlyData


def make_numpy_func(expr, vars=("x", "y")):
    # Define symbols for all variable names
    symbols = sp.symbols(vars)
    # Parse the expression safely
    sym_expr = sp.sympify(expr)
    # Convert symbolic expression to NumPy function
    f = sp.lambdify(symbols, sym_expr, modules=["numpy"])
    return f


def make_numpy_deri(expr, vars=("x", "y")):
    symbols = sp.symbols(vars)
    sym_expr = sp.sympify(expr)
    dx = sp.diff(sym_expr, symbols[0])
    dy = sp.diff(sym_expr, symbols[1])
    fdx = sp.lambdify(symbols, dx, modules=["numpy"])
    fdy = sp.lambdify(symbols, dy, modules=["numpy"])
    return fdx, fdy


def load_texture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    img_data = img.tobytes()

    return img_data, *(img.size)


def vertices_to_coords(vertices):
    return np.array([o.vertex.flatten() for o in vertices], dtype=np.float32)


def vertices_to_colors(vertices):
    return np.array([o.color.flatten() for o in vertices], dtype=np.float32)


def generate_gradient_colors(vertices, gradient_mode, start_color, end_color):
    """
    Generate gradient colors for vertices based on gradient mode.

    Args:
        vertices: List of Vertex objects or numpy array of coordinates
        gradient_mode: GradientMode enum value
        start_color: tuple (r, g, b) for start color (0-1 range)
        end_color: tuple (r, g, b) for end color (0-1 range)

    Returns:
        numpy array of colors (n_vertices, 3)
    """
    from config import GradientMode

    # Extract coordinates
    if hasattr(vertices[0], "vertex"):
        coords = np.array([v.vertex for v in vertices], dtype=np.float32)
    else:
        coords = np.array(vertices, dtype=np.float32)

    n_vertices = len(coords)
    colors = np.zeros((n_vertices, 3), dtype=np.float32)

    start_color = np.array(start_color, dtype=np.float32)
    end_color = np.array(end_color, dtype=np.float32)

    if gradient_mode == GradientMode.NONE or gradient_mode is None:
        # Return uniform color (start_color)
        colors[:] = start_color

    elif gradient_mode == GradientMode.LINEAR_X:
        # Gradient along X axis
        x_coords = coords[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        if x_max != x_min:
            t = (x_coords - x_min) / (x_max - x_min)
        else:
            t = np.zeros_like(x_coords)
        colors = start_color + np.outer(t, (end_color - start_color))

    elif gradient_mode == GradientMode.LINEAR_Y:
        # Gradient along Y axis
        y_coords = coords[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        if y_max != y_min:
            t = (y_coords - y_min) / (y_max - y_min)
        else:
            t = np.zeros_like(y_coords)
        colors = start_color + np.outer(t, (end_color - start_color))

    elif gradient_mode == GradientMode.LINEAR_Z:
        # Gradient along Z axis
        z_coords = coords[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        if z_max != z_min:
            t = (z_coords - z_min) / (z_max - z_min)
        else:
            t = np.zeros_like(z_coords)
        colors = start_color + np.outer(t, (end_color - start_color))

    elif gradient_mode == GradientMode.RADIAL:
        # Radial gradient from center
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        d_min, d_max = distances.min(), distances.max()
        if d_max != d_min:
            t = (distances - d_min) / (d_max - d_min)
        else:
            t = np.zeros_like(distances)
        colors = start_color + np.outer(t, (end_color - start_color))

    elif gradient_mode == GradientMode.DIAGONAL:
        # Diagonal gradient (X + Y + Z)
        diagonal = coords[:, 0] + coords[:, 1] + coords[:, 2]
        d_min, d_max = diagonal.min(), diagonal.max()
        if d_max != d_min:
            t = (diagonal - d_min) / (d_max - d_min)
        else:
            t = np.zeros_like(diagonal)
        colors = start_color + np.outer(t, (end_color - start_color))

    elif gradient_mode == GradientMode.RAINBOW:
        # Rainbow gradient (hue cycle)
        # Use Y coordinate for rainbow
        y_coords = coords[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        if y_max != y_min:
            t = (y_coords - y_min) / (y_max - y_min)
        else:
            t = np.zeros_like(y_coords)

        # Convert HSV to RGB (H varies, S=1, V=1)
        for i, hue in enumerate(t):
            h = hue * 6.0  # Hue 0-6
            c = 1.0
            x = c * (1.0 - abs((h % 2) - 1.0))
            m = 0.0

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors[i] = [r + m, g + m, b + m]

    return colors


def load_ply(path):
    """Load a PLY file and return mesh data in the same format as pyassimp."""
    meshes = []
    ply_data = PlyData.read(path)

    # Extract vertex data
    vertex_data = ply_data["vertex"]
    vertices = np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float32)

    # Extract normals (if available)
    if "nx" in vertex_data and "ny" in vertex_data and "nz" in vertex_data:
        normals = np.column_stack(
            [vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]]
        ).astype(np.float32)
    else:
        # Create zero normals if not available
        normals = np.zeros_like(vertices, dtype=np.float32)

    # Extract texture coordinates (if available)
    if "u" in vertex_data and "v" in vertex_data:
        tex_coords = np.column_stack([vertex_data["u"], vertex_data["v"]]).astype(
            np.float32
        )
    elif "s" in vertex_data and "t" in vertex_data:
        tex_coords = np.column_stack([vertex_data["s"], vertex_data["t"]]).astype(
            np.float32
        )
    else:
        # Create zero texture coordinates if not available
        tex_coords = np.zeros((len(vertices), 2), dtype=np.float32)

    # Extract face indices
    if "face" in ply_data:
        face_data = ply_data["face"]
        indices = np.array(
            [face for face in face_data["vertex_indices"]], dtype=np.uint32
        ).flatten()
    else:
        # No faces, create empty indices
        indices = np.array([], dtype=np.uint32)

    meshes.append(
        {
            "vertices": vertices,
            "normals": normals,
            "tex_coords": tex_coords,
            "indices": indices,
        }
    )

    return meshes


def load_mtl(path):
    """Load MTL file and return material information"""
    materials = {}
    current_material = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "newmtl":
                # New material definition
                current_material = parts[1]
                materials[current_material] = {
                    "diffuse_color": [0.8, 0.8, 0.8],
                    "diffuse_texture": None,
                    "specular_color": [1.0, 1.0, 1.0],
                    "specular_texture": None,
                    "normal_texture": None,
                }
            elif current_material:
                if parts[0] == "Kd":
                    # Diffuse color
                    materials[current_material]["diffuse_color"] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                elif parts[0] == "Ks":
                    # Specular color
                    materials[current_material]["specular_color"] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                elif parts[0] == "map_Kd":
                    # Diffuse texture map
                    materials[current_material]["diffuse_texture"] = parts[1]
                elif parts[0] == "map_Ks":
                    # Specular texture map
                    materials[current_material]["specular_texture"] = parts[1]
                elif parts[0] == "map_Bump" or parts[0] == "bump":
                    # Normal/bump map
                    materials[current_material]["normal_texture"] = parts[1]

    return materials


def load_obj(path):
    """Load OBJ file format - matches pyassimp output format"""
    raw_vertices = []
    raw_normals = []
    raw_tex_coords = []
    faces = []  # List of face tuples: (v_idx, vt_idx, vn_idx)
    mtl_file = None
    current_material = None
    material_faces = {}  # Track which faces use which material

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":
                # Vertex position
                raw_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vn":
                # Vertex normal
                raw_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt":
                # Texture coordinate
                raw_tex_coords.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "mtllib":
                # Material library file
                mtl_file = parts[1]
            elif parts[0] == "usemtl":
                # Use material
                current_material = parts[1]
                if current_material not in material_faces:
                    material_faces[current_material] = []
            elif parts[0] == "f":
                # Face - parse all vertex indices
                face = []
                for i in range(1, len(parts)):
                    indices = parts[i].split("/")
                    # Vertex index (1-based, convert to 0-based)
                    v_idx = int(indices[0]) - 1
                    # Texture coordinate index (optional)
                    vt_idx = (
                        int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                    )
                    # Normal index (optional)
                    vn_idx = (
                        int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
                    )
                    face.append((v_idx, vt_idx, vn_idx))

                # Triangulate if needed
                num_verts = len(face)
                if num_verts == 3:
                    triangle = face
                    if current_material:
                        material_faces[current_material].append(triangle)
                    else:
                        faces.append(triangle)
                elif num_verts > 3:
                    # Fan triangulation
                    for i in range(1, num_verts - 1):
                        triangle = [face[0], face[i], face[i + 1]]
                        if current_material:
                            material_faces[current_material].append(triangle)
                        else:
                            faces.append(triangle)

    raw_vertices = np.array(raw_vertices, dtype=np.float32)
    raw_normals = np.array(raw_normals, dtype=np.float32) if raw_normals else None
    raw_tex_coords = (
        np.array(raw_tex_coords, dtype=np.float32) if raw_tex_coords else None
    )

    # Load materials if MTL file is specified
    materials = {}
    if mtl_file:
        from pathlib import Path

        mtl_path = Path(path).parent / mtl_file
        if mtl_path.exists():
            materials = load_mtl(str(mtl_path))

    # Build separate meshes for each material group
    meshes = []

    # Process faces with materials (create one mesh per material)
    for mat_name, mat_faces in material_faces.items():
        if not mat_faces:
            continue

        unique_verts = {}
        out_vertices = []
        out_normals = []
        out_tex_coords = []
        out_indices = []

        for face in mat_faces:
            for v_idx, vt_idx, vn_idx in face:
                key = (v_idx, vt_idx, vn_idx)
                if key not in unique_verts:
                    new_idx = len(out_vertices)
                    unique_verts[key] = new_idx

                    # Add vertex position
                    out_vertices.append(raw_vertices[v_idx])

                    # Add texture coordinate
                    if raw_tex_coords is not None and vt_idx >= 0:
                        out_tex_coords.append(raw_tex_coords[vt_idx])
                    else:
                        out_tex_coords.append([0.0, 0.0])

                    # Add normal
                    if raw_normals is not None and vn_idx >= 0:
                        out_normals.append(raw_normals[vn_idx])
                    else:
                        out_normals.append([0.0, 0.0, 0.0])

                out_indices.append(unique_verts[key])

        out_vertices = np.array(out_vertices, dtype=np.float32)
        out_normals = np.array(out_normals, dtype=np.float32)
        out_tex_coords = np.array(out_tex_coords, dtype=np.float32)
        out_indices = np.array(out_indices, dtype=np.uint32)

        # Generate normals if not present in the file
        if raw_normals is None or len(raw_normals) == 0:
            out_normals = np.zeros_like(out_vertices)
            if len(out_indices) >= 3:
                indices_reshaped = out_indices.reshape(-1, 3)
                for face_indices in indices_reshaped:
                    v0, v1, v2 = out_vertices[face_indices]
                    normal = np.cross(v1 - v0, v2 - v0)
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 0:
                        normal = normal / norm_len
                    out_normals[face_indices] += normal
                # Normalize accumulated normals
                norms = np.linalg.norm(out_normals, axis=1, keepdims=True)
                norms[norms == 0] = 1
                out_normals = out_normals / norms

        # Get material info for this mesh
        mat_info = materials.get(mat_name, {})

        meshes.append(
            {
                "name": mat_name,
                "vertices": out_vertices,
                "normals": out_normals,
                "tex_coords": out_tex_coords,
                "indices": out_indices,
                "materials": {mat_name: mat_info},
                "material_name": mat_name,
                "position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "rotation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            }
        )

    # Process faces without materials (if any)
    if faces:
        unique_verts = {}
        out_vertices = []
        out_normals = []
        out_tex_coords = []
        out_indices = []

        for face in faces:
            for v_idx, vt_idx, vn_idx in face:
                key = (v_idx, vt_idx, vn_idx)
                if key not in unique_verts:
                    new_idx = len(out_vertices)
                    unique_verts[key] = new_idx

                    # Add vertex position
                    out_vertices.append(raw_vertices[v_idx])

                    # Add texture coordinate
                    if raw_tex_coords is not None and vt_idx >= 0:
                        out_tex_coords.append(raw_tex_coords[vt_idx])
                    else:
                        out_tex_coords.append([0.0, 0.0])

                    # Add normal
                    if raw_normals is not None and vn_idx >= 0:
                        out_normals.append(raw_normals[vn_idx])
                    else:
                        out_normals.append([0.0, 0.0, 0.0])

                out_indices.append(unique_verts[key])

        out_vertices = np.array(out_vertices, dtype=np.float32)
        out_normals = np.array(out_normals, dtype=np.float32)
        out_tex_coords = np.array(out_tex_coords, dtype=np.float32)
        out_indices = np.array(out_indices, dtype=np.uint32)

        # Generate normals if not present in the file
        if raw_normals is None or len(raw_normals) == 0:
            out_normals = np.zeros_like(out_vertices)
            if len(out_indices) >= 3:
                indices_reshaped = out_indices.reshape(-1, 3)
                for face_indices in indices_reshaped:
                    v0, v1, v2 = out_vertices[face_indices]
                    normal = np.cross(v1 - v0, v2 - v0)
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 0:
                        normal = normal / norm_len
                    out_normals[face_indices] += normal
                # Normalize accumulated normals
                norms = np.linalg.norm(out_normals, axis=1, keepdims=True)
                norms[norms == 0] = 1
                out_normals = out_normals / norms

        meshes.append(
            {
                "name": "default",
                "vertices": out_vertices,
                "normals": out_normals,
                "tex_coords": out_tex_coords,
                "indices": out_indices,
                "materials": materials,
                "material_name": None,
                "position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "rotation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            }
        )

    return meshes


def load_model(path):
    """Load 3D model from PLY, OBJ, GLTF, GLB, FBX, or other formats supported by pyassimp"""
    ext = path.lower().split(".")[-1]

    if ext == "ply":
        return load_ply(path)
    elif ext == "obj":
        return load_obj(path)
    else:
        # Use pyassimp for other formats (.gltf, .glb, .fbx, .3ds, .blend, etc.)
        meshes = []
        materials = {}

        try:
            with pyassimp.load(path) as scene:
                # Extract materials from the scene
                if scene.materials:
                    for idx, mat in enumerate(scene.materials):
                        mat_name = f"material_{idx}"
                        mat_props = mat.properties

                        materials[mat_name] = {
                            "diffuse_color": list(
                                mat_props.get(("COLOR_DIFFUSE", 0), [0.8, 0.8, 0.8])[:3]
                            ),
                            "specular_color": list(
                                mat_props.get(("COLOR_SPECULAR", 0), [1.0, 1.0, 1.0])[
                                    :3
                                ]
                            ),
                            "diffuse_texture": None,
                            "specular_texture": None,
                            "normal_texture": None,
                        }

                        # Extract texture paths if available
                        try:
                            if ("file", 1) in mat_props:  # Diffuse texture
                                tex_path = mat_props[("file", 1)]
                                # Convert to string if it's an array
                                if hasattr(tex_path, "__iter__") and not isinstance(
                                    tex_path, str
                                ):
                                    tex_path = (
                                        str(tex_path[0]) if len(tex_path) > 0 else None
                                    )
                                materials[mat_name]["diffuse_texture"] = tex_path
                        except (KeyError, IndexError, TypeError):
                            pass  # No texture found

                # Helper to decompose a 4x4 transformation matrix into translation, rotation (Euler degrees), and scale
                def decompose_matrix(mat):
                    try:
                        M = np.array(mat, dtype=np.float64)
                        # Ensure shape is (4,4)
                        if M.shape != (4, 4):
                            M = M.reshape((4, 4))

                        # Translation is in the last column (assimp stores row-major 4x4)
                        t = np.array([M[0, 3], M[1, 3], M[2, 3]], dtype=np.float32)

                        # Scale is the length of the basis vectors (first 3 columns)
                        sx = float(np.linalg.norm(M[0:3, 0]))
                        sy = float(np.linalg.norm(M[0:3, 1]))
                        sz = float(np.linalg.norm(M[0:3, 2]))
                        s = np.array([sx, sy, sz], dtype=np.float32)

                        # Build a rotation matrix by normalizing the basis vectors
                        R = np.zeros((3, 3), dtype=np.float64)
                        if sx != 0:
                            R[:, 0] = M[0:3, 0] / sx
                        else:
                            R[:, 0] = M[0:3, 0]
                        if sy != 0:
                            R[:, 1] = M[0:3, 1] / sy
                        else:
                            R[:, 1] = M[0:3, 1]
                        if sz != 0:
                            R[:, 2] = M[0:3, 2] / sz
                        else:
                            R[:, 2] = M[0:3, 2]

                        # Convert rotation matrix R to Euler angles (X, Y, Z) in degrees
                        # Use a safe asin for the Y angle
                        import math

                        # Following convention: R = Rz * Ry * Rx
                        # Compute angles
                        sy_val = -R[2, 0]
                        if sy_val < 1:
                            if sy_val > -1:
                                rx = math.atan2(R[2, 1], R[2, 2])
                                ry = math.asin(sy_val)
                                rz = math.atan2(R[1, 0], R[0, 0])
                            else:
                                # sy_val <= -1
                                rx = -math.atan2(-R[1, 2], R[1, 1])
                                ry = -math.pi / 2
                                rz = 0.0
                        else:
                            # sy_val >= 1
                            rx = math.atan2(-R[1, 2], R[1, 1])
                            ry = math.pi / 2
                            rz = 0.0

                        rot = np.degrees(np.array([rx, ry, rz], dtype=np.float32))
                        return t, rot, s
                    except Exception:
                        return (
                            np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 1.0, 1.0], dtype=np.float32),
                        )

                # Build mesh index to node/world transform mapping
                mesh_nodes = {}

                def traverse_nodes(node, parent_matrix=None):
                    """Traverse scene graph to find mesh assignments and accumulate transforms"""
                    if parent_matrix is None:
                        parent_matrix = np.identity(4, dtype=np.float64)

                    # Get node local transform and compute world transform
                    try:
                        node_mat = np.array(node.transformation, dtype=np.float64)
                        if node_mat.shape != (4, 4):
                            node_mat = node_mat.reshape((4, 4))
                    except Exception:
                        node_mat = np.identity(4, dtype=np.float64)

                    world_mat = parent_matrix @ node_mat

                    # Decompose world matrix to position, rotation, scale
                    node_pos, node_rot, node_scale = decompose_matrix(world_mat)

                    # node.meshes may contain indices or mesh references; normalize to integer indices
                    for mesh_ref in node.meshes:
                        mesh_idx = None
                        try:
                            if isinstance(mesh_ref, (int,)) or hasattr(
                                mesh_ref, "__int__"
                            ):
                                mesh_idx = int(mesh_ref)
                            else:
                                # Try to find mesh by identity in scene.meshes
                                for i, sm in enumerate(scene.meshes):
                                    if sm is mesh_ref:
                                        mesh_idx = i
                                        break
                                # Fallback: try attribute 'index'
                                if mesh_idx is None and hasattr(mesh_ref, "index"):
                                    mesh_idx = int(getattr(mesh_ref, "index"))
                        except Exception:
                            mesh_idx = None

                        if mesh_idx is None:
                            continue

                        mesh_nodes[int(mesh_idx)] = {
                            "name": node.name if node.name else f"mesh_{mesh_idx}",
                            "position": node_pos,
                            "rotation": node_rot,
                            "scale": node_scale,
                        }

                    # Recurse to children with accumulated world matrix
                    for child in node.children:
                        traverse_nodes(child, world_mat)

                # Start traversal from root
                if scene.rootnode:
                    traverse_nodes(scene.rootnode)

                # Process each mesh in the scene
                for mesh_idx, mesh in enumerate(scene.meshes):
                    vertices = np.array(mesh.vertices, dtype=np.float32)
                    # Safely handle normals which may be None or empty
                    if getattr(mesh, "normals", None) is not None:
                        normals = np.array(mesh.normals, dtype=np.float32)
                    else:
                        normals = np.zeros((len(vertices), 3), dtype=np.float32)

                    # Handle texture coordinates (avoid ambiguous truth-value on numpy arrays)
                    if (
                        getattr(mesh, "texturecoords", None) is not None
                        and len(mesh.texturecoords) > 0
                    ):
                        tex_coords = np.array(
                            mesh.texturecoords[0][:, :2], dtype=np.float32
                        )
                    else:
                        tex_coords = np.zeros((len(vertices), 2), dtype=np.float32)

                    # Handle indices
                    indices = np.array(mesh.faces, dtype=np.uint32).flatten()

                    # Get transform info from mesh_nodes mapping
                    node_info = mesh_nodes.get(
                        mesh_idx,
                        {
                            "name": f"mesh_{mesh_idx}",
                            "position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            "rotation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                        },
                    )

                    meshes.append(
                        {
                            "vertices": vertices,
                            "normals": normals,
                            "tex_coords": tex_coords,
                            "indices": indices,
                            "materials": materials,
                            "material_index": (
                                mesh.materialindex
                                if hasattr(mesh, "materialindex")
                                else 0
                            ),
                            "name": node_info["name"],
                            "position": node_info["position"],
                            "rotation": node_info["rotation"],
                            "scale": node_info["scale"],
                        }
                    )
        except Exception as e:
            # Log detailed error for debugging
            error_msg = f"Error loading model '{path}' with pyassimp: {e}"
            print(error_msg)
            print(f"File extension: .{ext}")
            print(
                f"Supported formats via pyassimp: .gltf, .glb, .fbx, .3ds, .dae, .blend, .obj, .ply"
            )
            print(
                "Note: Some formats may not be fully supported. Try exporting to .gltf, .glb, .obj or .fbx for best compatibility."
            )

            # Attempt graceful fallback: if there is a same-named .obj or .ply next to the file, try loading that
            try:
                from pathlib import Path

                p = Path(path)
                sibling_obj = p.with_suffix(".obj")
                sibling_ply = p.with_suffix(".ply")

                if sibling_obj.exists():
                    print(
                        f"pyassimp failed; attempting to load fallback OBJ: {sibling_obj}"
                    )
                    return load_obj(str(sibling_obj))
                if sibling_ply.exists():
                    print(
                        f"pyassimp failed; attempting to load fallback PLY: {sibling_ply}"
                    )
                    return load_ply(str(sibling_ply))
            except Exception:
                pass

            # As a last resort, return an empty mesh list so the application can continue
            print(f"✗ Error loading asset: {error_msg}")
            return []

        return meshes
