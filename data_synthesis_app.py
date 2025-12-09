"""
Data Synthesis Application - Main entry point
"""

import sys
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
from OpenGL import GL
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

from rendering.renderer import Renderer
from rendering.camera import Camera
from config import ShadingModel
from synthesis.scene import SynthesisScene
from synthesis.rendering import (
    DepthRenderer,
    SegmentationRenderer,
    extract_object_masks,
    compute_bounding_box_2d,
)
from synthesis.export import COCOExporter, YOLOExporter, compute_bbox_from_projection
from synthesis.placement import sample_non_colliding_position


class Config:
    """Simple configuration class"""

    def __init__(self):
        self.width = 1280
        self.height = 720
        self.camera = type(
            "obj",
            (object,),
            {
                "position": np.array([0.0, 5.0, 10.0]),
                "front": np.array([0.0, 0.0, -1.0]),
                "up": np.array([0.0, 1.0, 0.0]),
                "right": np.array([1.0, 0.0, 0.0]),
                "yaw": -90.0,
                "pitch": -20.0,
                "fov": 45.0,
                "near_plane": 0.1,
                "far_plane": 100.0,
                "move_speed": 5.0,
                "sensitivity": 0.1,
            },
        )
        self.trackball = type(
            "obj",
            (object,),
            {
                "position": np.array([0.0, 0.0, 0.0]),
                "yaw": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "radians": False,
                "distance": 10.0,
                "pan_sensitivity": 0.01,
            },
        )
        self.cull_face = False


class DataSynthesisApp:
    """Main application for data synthesis"""

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.window = None
        self.imgui_impl = None

        # Initialize tkinter root (hidden)
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

        # Scene and rendering
        self.config = Config()
        self.config.width = width
        self.config.height = height

        # UI state
        self.object_path = ""
        self.object_class = "object"
        self.output_dir = "output"
        # Object selection
        self.selected_object_idx = -1
        self.selected_group_idx = -1
        self.selected_object_in_group_idx = -1
        # Candidate objects (for synthesis)
        # Each entry: { 'id': str, 'name': str, 'ref': SynthesisObject }
        self.candidate_objects = []
        self.image_counter = 0

        # Shading model
        self.current_shading = 1  # 0=Normal, 1=Phong, 2=Gouraud

        # Transformation sliders for selected object
        self.transform_position = [0.0, 0.0, 0.0]
        self.transform_rotation = [0.0, 0.0, 0.0]
        self.transform_scale = 1.0  # Input state
        self.mouse_pressed = False
        self.right_mouse_pressed = False
        self.middle_mouse_pressed = False
        self.last_mouse_pos = (0, 0)

        # Camera FPS mode: False = disabled (normal mouse), True = enabled (locked mouse)
        self.use_camera_mode = False

        # Camera sensitivity settings (reduced move speed, increased look sensitivity)
        self.camera_move_speed = 1.5
        self.camera_look_sensitivity = 0.35

        # Keyboard state for Camera mode
        self.keys_down = set()

        # Assets library
        self.available_assets = []  # List of asset names from ./assets folder
        self.selected_asset_idx = 0
        self._scan_assets_folder()

        self._scan_assets_folder()

        # Normalization box size (max dimension in world units to fit loaded objects)
        # Objects/groups will be scaled so their largest side fits within this size
        self.load_normalize_size = 6.0

        # Animation recording
        self.is_recording_animation = False
        self.current_animation_keypoints = []
        self.recorded_animations = []  # List of {"name": str, "keypoints": list}
        self.selected_animation_idx = -1
        self.frames_per_capture = 30
        self.animation_name = "Animation_1"

        self._init_glfw()

        # Initialize renderer AFTER OpenGL context is created
        self.renderer = Renderer(self.config)
        self.synthesis_scene = SynthesisScene()

        # Special renderers
        self.depth_renderer = DepthRenderer(width, height)
        self.seg_renderer = SegmentationRenderer(width, height)

        # Light parameters (for randomization)
        self.light_position = np.array([5.0, 10.0, 5.0])  # Default light position
        self.light_color = np.array(
            [1.0, 1.0, 1.0]
        )  # Default white light (strength = 1.0)
        self.light_strength = 1.0

        # Connect renderer to app
        self.renderer.app = self
        self.renderer.use_trackball = False

        # Set initial scene
        self.renderer.set_scene(self.synthesis_scene.get_root())

    def _scan_assets_folder(self):
        """Scan ./assets folder and detect available models"""
        assets_path = Path("assets")
        self.available_assets = []

        if not assets_path.exists():
            print("Warning: ./assets folder not found")
            return

        # Scan each subfolder in assets
        for item in sorted(assets_path.iterdir()):
            if item.is_dir():
                self.available_assets.append(item.name)

        print(f"Found {len(self.available_assets)} assets: {self.available_assets}")

    def _load_asset_from_library(self, asset_name: str):
        """Load a model from the assets library with auto-detection of textures"""
        asset_path = Path("assets") / asset_name

        if not asset_path.exists():
            print(f"Asset folder not found: {asset_path}")
            return

        # Check for .obj file directly in asset root (legacy support)
        obj_files = list(asset_path.glob("*.obj"))
        if obj_files:
            obj_file = obj_files[0]
            print(f"Loading OBJ model (legacy): {obj_file}")

            try:
                # Add the object
                obj = self.synthesis_scene.add_object(str(obj_file), asset_name)

                # Try to apply material textures (searches nearby texture dirs)
                try:
                    obj.shape.apply_material_textures()
                except Exception as e:
                    print(f"Warning: apply_material_textures failed: {e}")

                # Auto-detect textures in textures/ subfolder (same as FBX/GLTF)
                textures_path = asset_path / "textures"
                if textures_path.exists():
                    print(f"Scanning for textures in: {textures_path}")
                    texture_files = (
                        list(textures_path.glob("*.png"))
                        + list(textures_path.glob("*.jpg"))
                        + list(textures_path.glob("*.jpeg"))
                    )
                    for texture_file in texture_files:
                        try:
                            key = texture_file.stem
                            if key not in obj.shape.textures:
                                print(f"Auto-loading texture: {texture_file}")
                                tex_id = obj.shape.load_texture(str(texture_file), key)
                                if tex_id and key not in obj.shape.textures:
                                    from graphics.texture import Texture2D

                                    obj.shape.textures[key] = Texture2D.from_id(tex_id)
                                obj.shape.texture_enabled = True
                                print(
                                    f"✓ Applied texture {texture_file.name} to {obj.name}"
                                )
                        except Exception as tex_error:
                            print(f"Note: Texture loading skipped: {tex_error}")

                # Normalize single object to fit within the invisible box
                try:
                    bb = obj.get_world_bounding_box()
                    size = bb["max"] - bb["min"]
                    max_dim = float(np.max(size)) if np.any(size) else 0.0
                    if max_dim > 0.0:
                        desired = float(self.load_normalize_size)
                        scale_factor = desired / max_dim
                        obj.scale_factor *= scale_factor
                        center = bb["center"]
                        obj.position = -center * scale_factor
                        print(
                            f"  Normalized object '{obj.name}': scale={scale_factor:.4g}, center={center}"
                        )
                except Exception as e:
                    print(f"  Warning: failed to normalize object '{obj.name}': {e}")

                # Fallback: auto-load any direct texture files in the asset folder
                if not textures_path.exists():
                    jpg_files = list(asset_path.glob("*.jpg")) + list(
                        asset_path.glob("*.png")
                    )
                    for texture_file in jpg_files:
                        try:
                            key = texture_file.stem
                            if key not in obj.shape.textures:
                                print(f"Auto-loading texture: {texture_file}")
                                obj.shape.load_texture(str(texture_file), key)
                                obj.shape.texture_enabled = True
                        except Exception as tex_error:
                            print(f"Note: Texture loading skipped: {tex_error}")

                self.renderer.set_scene(self.synthesis_scene.get_root())
                print(f"✓ Successfully loaded asset: {asset_name}")
                return
            except Exception as e:
                print(f"✗ Error loading OBJ asset: {e}")
                return

        # Check for 3D model files (FBX, GLTF, GLB) in source/ folder - load as object group
        source_path = asset_path / "source"
        if source_path.exists():
            # Check for various 3D model formats
            model_file = None
            model_type = None

            for ext, type_name in [
                ("*.obj", "OBJ"),
                ("*.fbx", "FBX"),
                ("*.gltf", "GLTF"),
                ("*.glb", "GLB"),
            ]:
                model_files = list(source_path.glob(ext))
                if model_files:
                    model_file = model_files[0]
                    model_type = type_name
                    break

            if model_file:
                print(f"Loading {model_type} model as group: {model_file}")

                try:
                    from utils.misc import load_model

                    # Load all meshes from the model file
                    meshes = load_model(str(model_file))
                    print(f"Found {len(meshes)} meshes in {model_type} file")

                    # Create object group
                    group = self.synthesis_scene.add_object_group(
                        asset_name, asset_name
                    )

                    # Create separate object for each mesh with preserved transforms
                    for mesh_idx, mesh_data in enumerate(meshes):
                        mesh_name = mesh_data.get("name", f"mesh_{mesh_idx}")
                        position = mesh_data.get("position", np.array([0.0, 0.0, 0.0]))
                        rotation = mesh_data.get("rotation", np.array([0.0, 0.0, 0.0]))
                        scale_vec = mesh_data.get("scale", np.array([1.0, 1.0, 1.0]))

                        # Create a temporary mesh file in memory by creating MeshShape directly
                        from shape.mesh_loader import MeshShape
                        from graphics.buffer import VAO
                        from shape.base import Part
                        from OpenGL import GL

                        # Get material color if available (for OBJ with MTL)
                        base_color = np.array([0.8, 0.2, 0.2], dtype=np.float32)
                        mat_name = mesh_data.get("material_name")
                        if mat_name and "materials" in mesh_data:
                            mat_info = mesh_data["materials"].get(mat_name, {})
                            diffuse_color = mat_info.get("diffuse_color")
                            if diffuse_color:
                                base_color = np.array(diffuse_color, dtype=np.float32)
                                print(
                                    f"    Using material '{mat_name}' color: {diffuse_color}"
                                )

                        # Create MeshShape with single mesh
                        shape = MeshShape.__new__(MeshShape)
                        shape.__dict__.update(
                            {
                                "mesh_path": Path(model_file),
                                "base_color": base_color,
                                "scale": 1.0,
                                "materials": mesh_data.get("materials", {}),
                                "textures": {},
                                "mesh_vertices": [],
                                "shapes": [],
                            }
                        )

                        # Initialize shader (call parent __init__ manually)
                        from shape.base import Shape

                        Shape.__init__(shape, None, None)

                        # Build VAO for this mesh
                        vertices = mesh_data["vertices"]
                        normals = mesh_data["normals"]
                        tex_coords = mesh_data["tex_coords"]
                        indices = mesh_data["indices"]

                        num_vertices = len(vertices)
                        colors = np.tile(shape.base_color, (num_vertices, 1))

                        vao = VAO()
                        vao.add_vbo(
                            0, vertices.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None
                        )
                        vao.add_vbo(
                            1, colors.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None
                        )
                        vao.add_vbo(
                            2, normals.flatten(), 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None
                        )
                        vao.add_vbo(
                            3,
                            tex_coords.flatten(),
                            2,
                            GL.GL_FLOAT,
                            GL.GL_FALSE,
                            0,
                            None,
                        )

                        if len(indices) > 0:
                            vao.add_ebo(indices)
                            part = Part(
                                vao, GL.GL_TRIANGLES, num_vertices, len(indices)
                            )
                        else:
                            part = Part(vao, GL.GL_TRIANGLES, num_vertices)

                        shape.shapes.append(part)
                        shape.mesh_vertices.append(vertices)

                        # Compute bounding box
                        shape.bbox_min = np.min(vertices, axis=0)
                        shape.bbox_max = np.max(vertices, axis=0)
                        shape.bbox_center = (shape.bbox_min + shape.bbox_max) / 2
                        shape.bbox_size = shape.bbox_max - shape.bbox_min

                        # Debug: log mesh geometry info
                        print(
                            f"    Mesh '{mesh_name}': {num_vertices} verts, {len(indices)} indices, bbox={shape.bbox_min} to {shape.bbox_max}"
                        )

                        # Create synthesis object with preserved transform
                        from synthesis.scene import SynthesisObject

                        obj = SynthesisObject(
                            name=f"{asset_name}_{mesh_name}",
                            shape=shape,
                            position=position,
                            rotation=rotation,
                            scale=float(np.mean(scale_vec)),  # Use average scale
                            class_id=group.class_id,
                        )

                        group.add_object(obj)
                        print(f"  Added mesh: {mesh_name} at pos={position}")

                    # Normalize group to fit within a predefined invisible box
                    try:
                        # Compute group's world AABB by combining object bboxes
                        mins = []
                        maxs = []
                        for o in group.objects:
                            bb = o.get_world_bounding_box()
                            mins.append(bb["min"])
                            maxs.append(bb["max"])

                        if mins and maxs:
                            global_min = np.min(np.vstack(mins), axis=0)
                            global_max = np.max(np.vstack(maxs), axis=0)
                            size = global_max - global_min
                            max_dim = float(np.max(size)) if np.any(size) else 0.0

                            if max_dim > 0.0:
                                desired = float(self.load_normalize_size)
                                scale_factor = desired / max_dim

                                # Apply uniform scale to group and translate to center at origin
                                group.scale_factor *= scale_factor
                                center = (global_min + global_max) / 2.0
                                group.position = -center * scale_factor
                                print(
                                    f"  Normalized group '{group.name}': scale={scale_factor:.4g}, center={center}"
                                )
                    except Exception as e:
                        print(
                            f"  Warning: failed to normalize group '{group.name}': {e}"
                        )
                    # Auto-detect and load textures from textures/ folder
                    textures_path = asset_path / "textures"
                    if textures_path.exists():
                        print(f"Scanning for textures in: {textures_path}")
                        texture_files = (
                            list(textures_path.glob("*.png"))
                            + list(textures_path.glob("*.jpg"))
                            + list(textures_path.glob("*.jpeg"))
                        )

                        if texture_files:
                            print(f"Found {len(texture_files)} texture files")

                            # Build a mapping of texture filenames (without extension) to file paths
                            texture_map = {}
                            for tex_file in texture_files:
                                tex_name_lower = tex_file.stem.lower()
                                texture_map[tex_name_lower] = tex_file

                            # Try to match textures to objects
                            for mesh_idx, obj in enumerate(group.objects):
                                # Extract the mesh name from the object name (format: assetname_meshname)
                                obj_parts = obj.name.split("_", 1)
                                mesh_name = (
                                    obj_parts[1] if len(obj_parts) > 1 else obj.name
                                )
                                mesh_name_lower = mesh_name.lower()

                                matched = False
                                best_match = None
                                best_score = 0

                                # First priority: Check if mesh has material info with texture reference
                                mesh_data = (
                                    meshes[mesh_idx] if mesh_idx < len(meshes) else None
                                )
                                if mesh_data:
                                    mat_name = mesh_data.get("material_name")
                                    if mat_name and "materials" in mesh_data:
                                        mat_info = mesh_data["materials"].get(
                                            mat_name, {}
                                        )
                                        diffuse_tex = mat_info.get("diffuse_texture")

                                        if diffuse_tex:
                                            # Extract just the filename from the MTL texture path
                                            import os

                                            tex_filename = os.path.splitext(
                                                os.path.basename(diffuse_tex)
                                            )[0].lower()

                                            # Look for this texture in the textures folder
                                            if tex_filename in texture_map:
                                                best_match = texture_map[tex_filename]
                                                matched = True
                                                print(
                                                    f"  ✓ MTL reference: {best_match.name} → {obj.name} (material: {mat_name})"
                                                )

                                # Second priority: Try exact mesh name match
                                if not matched and mesh_name_lower in texture_map:
                                    best_match = texture_map[mesh_name_lower]
                                    matched = True
                                    print(
                                        f"  ✓ Exact match: {best_match.name} → {obj.name}"
                                    )

                                # Third priority: Try material name match (for OBJ files)
                                if not matched and mesh_data:
                                    mat_name = mesh_data.get(
                                        "material_name", ""
                                    ).lower()
                                    if mat_name in texture_map:
                                        best_match = texture_map[mat_name]
                                        matched = True
                                        print(
                                            f"  ✓ Material name match: {best_match.name} → {obj.name}"
                                        )

                                # Fourth priority: Fuzzy matching
                                if not matched:
                                    # Try fuzzy matching: find texture with most matching words
                                    mesh_words = set(
                                        mesh_name_lower.replace("-", "_").split("_")
                                    )

                                    for tex_name_lower, tex_file in texture_map.items():
                                        tex_words = set(
                                            tex_name_lower.replace("-", "_").split("_")
                                        )
                                        # Count matching words
                                        common_words = mesh_words & tex_words
                                        score = len(common_words)

                                        if score > best_score:
                                            best_score = score
                                            best_match = tex_file

                                    # Also check if texture name is substring of mesh name or vice versa
                                    for tex_name_lower, tex_file in texture_map.items():
                                        if (
                                            tex_name_lower in mesh_name_lower
                                            or mesh_name_lower in tex_name_lower
                                        ):
                                            if (
                                                best_score < 2
                                            ):  # Substring match is better than 1 word match
                                                best_match = tex_file
                                                best_score = 2

                                    if best_match and best_score > 0:
                                        matched = True
                                        print(
                                            f"  ✓ Fuzzy match (score={best_score}): {best_match.name} → {obj.name}"
                                        )

                                # Apply the matched texture
                                if matched and best_match:
                                    try:
                                        texture_id = obj.shape.load_texture(
                                            str(best_match)
                                        )
                                        if texture_id:
                                            # Enable texture for this mesh (Unity-like behavior)
                                            obj.shape.texture_enabled = True
                                            from graphics.texture import Texture2D

                                            if not obj.shape.texture:
                                                # Wrap the existing GL texture id in a Texture2D helper
                                                obj.shape.texture = Texture2D.from_id(
                                                    texture_id
                                                )
                                            print(
                                                f"  ✓ Applied texture {best_match.name} to {obj.name}"
                                            )
                                    except Exception as e:
                                        print(
                                            f"  ✗ Failed to apply {best_match.name}: {e}"
                                        )

                                # If no match found, try first available texture as fallback
                                elif texture_files:
                                    try:
                                        texture_id = obj.shape.load_texture(
                                            str(texture_files[0])
                                        )
                                        if texture_id:
                                            obj.shape.texture_enabled = True
                                            from graphics.texture import Texture2D

                                            if not obj.shape.texture:
                                                obj.shape.texture = Texture2D.from_id(
                                                    texture_id
                                                )
                                            print(
                                                f"  ✓ Applied fallback texture {texture_files[0].name} to {obj.name}"
                                            )
                                    except Exception as e:
                                        print(
                                            f"  ✗ Failed to apply fallback texture: {e}"
                                        )

                    self.synthesis_scene._rebuild_scene()
                    self.renderer.set_scene(self.synthesis_scene.get_root())

                    # Debug: verify scene graph
                    root = self.synthesis_scene.get_root()
                    print(f"  Scene root has {len(root.children)} top-level nodes")
                    for child in root.children:
                        print(f"    Node: {child.name} (type: {type(child).__name__})")

                    # Reset camera to look at the loaded object group
                    if group.objects:
                        # Compute group bounding box (filter out extreme outliers)
                        all_verts = []
                        for obj in group.objects:
                            if hasattr(obj.shape, "mesh_vertices"):
                                for mesh_verts in obj.shape.mesh_vertices:
                                    # Skip meshes with extreme coordinates (likely scale errors)
                                    mesh_extent = np.max(np.abs(mesh_verts))
                                    if mesh_extent < 50:  # Reasonable threshold
                                        all_verts.append(mesh_verts)

                        if all_verts:
                            all_verts = np.vstack(all_verts)
                            bbox_min = np.min(all_verts, axis=0)
                            bbox_max = np.max(all_verts, axis=0)
                            bbox_center = (bbox_min + bbox_max) / 2
                            bbox_size = bbox_max - bbox_min
                            max_dim = np.max(bbox_size)

                            # Position camera to view the entire object
                            # Use reasonable limits to avoid extreme camera positions
                            camera_distance = np.clip(max_dim * 2.5, 5.0, 50.0)
                            camera_height = np.clip(max_dim * 0.3, 1.0, 10.0)

                            self.renderer.camera.position = bbox_center + np.array(
                                [0, camera_height, camera_distance]
                            )
                            self.renderer.camera.front = np.array([0.0, 0.0, -1.0])
                            self.renderer.camera.yaw = -90.0
                            self.renderer.camera.pitch = -15.0
                            self.renderer.camera._recalculate_basis()
                            print(
                                f"  Camera positioned at {self.renderer.camera.position} (distance={camera_distance:.1f})"
                            )
                            print(
                                f"  Scene bbox: {bbox_min} to {bbox_max}, center={bbox_center}"
                            )

                    print(
                        f"✓ Successfully loaded {model_type} group: {asset_name} ({len(group.objects)} objects)"
                    )
                    return
                except Exception as e:
                    import traceback

                    print(f"✗ Error loading {model_type} asset: {e}")
                    traceback.print_exc()
                    return

        print(f"No supported model files found in {asset_path}")

    def _init_glfw(self):
        """Initialize GLFW window"""
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        # Create window
        self.window = glfw.create_window(
            self.width, self.height, "Data Synthesis Tool", None, None
        )

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # V-sync
        self._init_imgui()

        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

    def _init_imgui(self):
        """Initialize ImGui"""
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.window)

    def _on_resize(self, window, width, height):
        """Handle window resize"""
        self.width = width
        self.height = height
        GL.glViewport(0, 0, width, height)

        # Resize depth and segmentation renderers
        self.depth_renderer = DepthRenderer(width, height)
        self.seg_renderer = SegmentationRenderer(width, height)

    def _on_mouse_button(self, window, button, action, mods):
        """Handle mouse button events"""
        # print(f"Press: {button}")
        if imgui.get_io().want_capture_mouse:
            return

        if (
            button in [glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT]
            and action == glfw.PRESS
        ):
            # Store initial mouse position when button is pressed
            x_pos, y_pos = glfw.get_cursor_pos(self.window)
            # Invert Y coordinate for trackball (OpenGL convention)
            self.last_mouse_pos = (x_pos, self.height - y_pos)

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.right_mouse_pressed = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.middle_mouse_pressed = action == glfw.PRESS

    def _on_mouse_move(self, window, xpos, ypos):
        """Handle mouse movement"""
        # Only check ImGui capture if no button is pressed
        if not self.mouse_pressed and not self.right_mouse_pressed:
            if imgui.get_io().want_capture_mouse:
                self.last_mouse_pos = (xpos, self.height - ypos)
                return

        current_pos = (xpos, self.height - ypos)
        dx = xpos - self.last_mouse_pos[0]
        dy = (self.height - ypos) - self.last_mouse_pos[1]

        # Object editing mode: rotate object on left-drag (standalone or group object)
        selected_obj = None

        if self.selected_object_idx >= 0 and self.selected_object_idx < len(
            self.synthesis_scene.objects
        ):
            selected_obj = self.synthesis_scene.objects[self.selected_object_idx]
        elif self.selected_group_idx >= 0 and self.selected_object_in_group_idx >= 0:
            group = self.synthesis_scene.object_groups[self.selected_group_idx]
            if self.selected_object_in_group_idx < len(group.objects):
                selected_obj = group.objects[self.selected_object_in_group_idx]
        elif self.selected_group_idx >= 0 and self.selected_object_in_group_idx < 0:
            selected_obj = self.synthesis_scene.object_groups[self.selected_group_idx]

        if selected_obj and self.mouse_pressed:
            # Rotate around Y (horizontal drag) and X (vertical drag)
            selected_obj.rotation[1] += dx * 0.5
            selected_obj.rotation[0] += dy * 0.5
            self.transform_rotation = list(selected_obj.rotation)
            self.synthesis_scene._rebuild_scene()
            self.renderer.set_scene(self.synthesis_scene.get_root())
        # Camera mode: look around on mouse move (no button needed)
        elif (
            self.use_camera_mode
            and not self.mouse_pressed
            and not self.right_mouse_pressed
        ):
            # Free look in camera mode with configurable sensitivity
            # Fix X-axis flip by negating dx
            scaled_dx = -dx * self.camera_look_sensitivity
            scaled_dy = dy * self.camera_look_sensitivity
            scaled_last_pos = (
                self.last_mouse_pos[0] - scaled_dx + dx,
                self.last_mouse_pos[1] - scaled_dy + dy,
            )
            self.renderer.rotate_camera(scaled_last_pos, current_pos)

        self.last_mouse_pos = current_pos

    def _on_scroll(self, window, xoffset, yoffset):
        """Handle mouse scroll events"""
        if imgui.get_io().want_capture_mouse:
            return

        # Object editing: scale on scroll (allow scaling to 0) - works for standalone and group objects
        selected_obj = None

        if self.selected_object_idx >= 0 and self.selected_object_idx < len(
            self.synthesis_scene.objects
        ):
            selected_obj = self.synthesis_scene.objects[self.selected_object_idx]
        elif self.selected_group_idx >= 0 and self.selected_object_in_group_idx >= 0:
            group = self.synthesis_scene.object_groups[self.selected_group_idx]
            if self.selected_object_in_group_idx < len(group.objects):
                selected_obj = group.objects[self.selected_object_in_group_idx]
        elif self.selected_group_idx >= 0 and self.selected_object_in_group_idx < 0:
            selected_obj = self.synthesis_scene.object_groups[self.selected_group_idx]

        if selected_obj:
            selected_obj.scale_factor = max(
                0.0, selected_obj.scale_factor + yoffset * 0.1
            )
            self.transform_scale = selected_obj.scale_factor
            self.synthesis_scene._rebuild_scene()
            self.renderer.set_scene(self.synthesis_scene.get_root())
        # Camera mode: zoom by moving forward/backward
        elif self.use_camera_mode:
            from rendering.camera import CameraMovement

            speed = 2.0
            if yoffset > 0:
                self.renderer.move_camera(CameraMovement.FORWARD, speed)
            else:
                self.renderer.move_camera(CameraMovement.BACKWARD, speed)

    def _on_key(self, window, key, scancode, action, mods):
        """Handle keyboard events"""
        if imgui.get_io().want_capture_keyboard:
            return

        # Track key state for Camera mode and object editing
        if action == glfw.PRESS:
            self.keys_down.add(key)

            # Q key: Toggle FPS camera on/off
            if key == glfw.KEY_Q and self.selected_object_idx < 0:
                if self.use_camera_mode:
                    self._disable_camera_mode()
                else:
                    self._enable_camera_mode()

            # R key: Record keypoint for animation
            if (
                key == glfw.KEY_R
                and self.use_camera_mode
                and self.selected_object_idx < 0
            ):
                self._record_keypoint()

            # F key: Finish recording animation
            if (
                key == glfw.KEY_F
                and self.use_camera_mode
                and self.is_recording_animation
            ):
                self._finish_recording_animation()

            # Escape key: Cancel recording
            if (
                key == glfw.KEY_ESCAPE
                and self.use_camera_mode
                and self.is_recording_animation
            ):
                self._cancel_recording_animation()

            # N key: Start new animation
            if (
                key == glfw.KEY_N
                and self.use_camera_mode
                and not self.is_recording_animation
            ):
                self._start_new_animation()

            # I key: Switch to Normal shading
            if key == glfw.KEY_I:
                self.current_shading = 0
                self.renderer.set_shading_model(ShadingModel.NORMAL)
                print("Switched to Normal shading")

            # O key: Switch to Gouraud shading
            if key == glfw.KEY_O:
                self.current_shading = 2
                self.renderer.set_shading_model(ShadingModel.GOURAUD)
                print("Switched to Gouraud shading")

            # P key: Switch to Phong shading
            if key == glfw.KEY_P:
                self.current_shading = 1
                self.renderer.set_shading_model(ShadingModel.PHONG)
                print("Switched to Phong shading")

            # U key: Randomize light position and strength
            if key == glfw.KEY_U:
                # Random position in a reasonable range around the scene
                self.light_position = np.array(
                    [
                        np.random.uniform(-10.0, 10.0),  # X
                        np.random.uniform(5.0, 15.0),  # Y (keep above scene)
                        np.random.uniform(-10.0, 10.0),  # Z
                    ]
                )
                # Random strength between 0.5 and 1.5
                self.light_strength = np.random.uniform(0.5, 1.5)
                self.light_color = np.array([self.light_strength] * 3)
                print(
                    f"Light randomized: position={self.light_position}, strength={self.light_strength:.2f}"
                )
        elif action == glfw.RELEASE:
            self.keys_down.discard(key)

    def _process_keyboard_input(self, delta_time):
        """Process keyboard input for camera movement or object editing"""
        from rendering.camera import CameraMovement

        # Check if Shift is held
        shift_held = (
            glfw.KEY_LEFT_SHIFT in self.keys_down
            or glfw.KEY_RIGHT_SHIFT in self.keys_down
        )

        # Object editing mode: WASD moves object, Shift+WS moves vertically
        # Support both standalone objects and objects within groups
        selected_obj = None

        if self.selected_object_idx >= 0 and self.selected_object_idx < len(
            self.synthesis_scene.objects
        ):
            selected_obj = self.synthesis_scene.objects[self.selected_object_idx]
        elif (
            self.selected_group_idx >= 0
            and self.selected_group_idx < len(self.synthesis_scene.object_groups)
            and self.selected_object_in_group_idx >= 0
        ):
            # Object within a group
            group = self.synthesis_scene.object_groups[self.selected_group_idx]
            if self.selected_object_in_group_idx < len(group.objects):
                selected_obj = group.objects[self.selected_object_in_group_idx]
        elif (
            self.selected_group_idx >= 0
            and self.selected_group_idx < len(self.synthesis_scene.object_groups)
            and self.selected_object_in_group_idx < 0
        ):
            # Entire group selected (transform the group)
            selected_obj = self.synthesis_scene.object_groups[self.selected_group_idx]

        if selected_obj:
            speed = 6.0 * delta_time
            moved = False

            if shift_held:
                # Shift+W/S: move vertically (Y axis)
                if glfw.KEY_W in self.keys_down:
                    selected_obj.position[1] += speed
                    moved = True
                if glfw.KEY_S in self.keys_down:
                    selected_obj.position[1] -= speed
                    moved = True
            else:
                # WASD: move on horizontal plane (XZ)
                if glfw.KEY_W in self.keys_down:
                    selected_obj.position[2] -= speed  # Forward (negative Z)
                    moved = True
                if glfw.KEY_S in self.keys_down:
                    selected_obj.position[2] += speed  # Backward (positive Z)
                    moved = True
                if glfw.KEY_A in self.keys_down:
                    selected_obj.position[0] -= speed  # Left (negative X)
                    moved = True
                if glfw.KEY_D in self.keys_down:
                    selected_obj.position[0] += speed  # Right (positive X)
                    moved = True

            if moved:
                self.transform_position = list(selected_obj.position)
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

        # Camera mode: WASD moves camera (only when no object selected)
        # Space for up, Shift for down
        elif self.use_camera_mode:
            speed = self.camera_move_speed * delta_time

            if glfw.KEY_W in self.keys_down:
                self.renderer.move_camera(CameraMovement.FORWARD, speed)
            if glfw.KEY_S in self.keys_down:
                self.renderer.move_camera(CameraMovement.BACKWARD, speed)
            if glfw.KEY_A in self.keys_down:
                self.renderer.move_camera(CameraMovement.LEFT, speed)
            if glfw.KEY_D in self.keys_down:
                self.renderer.move_camera(CameraMovement.RIGHT, speed)
            if glfw.KEY_SPACE in self.keys_down:
                self.renderer.move_camera(CameraMovement.UP, speed)
            if shift_held:
                self.renderer.move_camera(CameraMovement.DOWN, speed)

    def _update_object_highlights(self):
        """Update visual highlights for selected objects"""
        # Unhighlight all objects first
        for obj in self.synthesis_scene.objects:
            if hasattr(obj.shape, "set_highlight"):
                obj.shape.set_highlight(False)

        for group in self.synthesis_scene.object_groups:
            for obj in group.objects:
                if hasattr(obj.shape, "set_highlight"):
                    obj.shape.set_highlight(False)

        # Highlight the selected object
        if self.selected_object_idx >= 0 and self.selected_object_idx < len(
            self.synthesis_scene.objects
        ):
            obj = self.synthesis_scene.objects[self.selected_object_idx]
            if hasattr(obj.shape, "set_highlight"):
                obj.shape.set_highlight(True)
        elif self.selected_group_idx >= 0 and self.selected_group_idx < len(
            self.synthesis_scene.object_groups
        ):
            group = self.synthesis_scene.object_groups[self.selected_group_idx]
            if (
                self.selected_object_in_group_idx >= 0
                and self.selected_object_in_group_idx < len(group.objects)
            ):
                obj = group.objects[self.selected_object_in_group_idx]
                if hasattr(obj.shape, "set_highlight"):
                    obj.shape.set_highlight(True)
            elif self.selected_object_in_group_idx < 0:
                # Highlight entire group
                for obj in group.objects:
                    if hasattr(obj.shape, "set_highlight"):
                        obj.shape.set_highlight(True)

    def _apply_lighting_to_scene(self):
        """Apply current light parameters to all objects in the scene"""
        # Apply to standalone objects
        for obj in self.synthesis_scene.objects:
            if hasattr(obj.shape, "lighting"):
                obj.shape.lighting(
                    self.light_color, self.light_position, self.renderer.camera.position
                )

        # Apply to group objects
        for group in self.synthesis_scene.object_groups:
            for obj in group.objects:
                if hasattr(obj.shape, "lighting"):
                    obj.shape.lighting(
                        self.light_color,
                        self.light_position,
                        self.renderer.camera.position,
                    )

    def _render_ui(self):
        """Render ImGui interface"""
        imgui.new_frame()

        # Main control panel
        imgui.set_next_window_position(10, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(400, 600, imgui.FIRST_USE_EVER)

        imgui.begin("Data Synthesis Control Panel", True)

        imgui.text("Data Synthesis Tool")
        imgui.separator()

        # Asset Loading Section
        if imgui.collapsing_header("Load Model", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text("Available Assets:")

            if len(self.available_assets) > 0:
                for asset_name in self.available_assets:
                    if imgui.button(f"Load {asset_name}"):
                        self._load_asset_from_library(asset_name)
            else:
                imgui.text_colored("No assets found in ./assets", 0.7, 0.7, 0.0, 1.0)

        imgui.separator()
        imgui.spacing()

        # Transform selected object / group (kept here while Objects list moved)
        selected_obj = None
        if self.selected_object_idx >= 0 and self.selected_object_idx < len(
            self.synthesis_scene.objects
        ):
            selected_obj = self.synthesis_scene.objects[self.selected_object_idx]
        elif (
            self.selected_group_idx >= 0
            and self.selected_group_idx < len(self.synthesis_scene.object_groups)
            and self.selected_object_in_group_idx >= 0
        ):
            group = self.synthesis_scene.object_groups[self.selected_group_idx]
            if self.selected_object_in_group_idx < len(group.objects):
                selected_obj = group.objects[self.selected_object_in_group_idx]
        elif (
            self.selected_group_idx >= 0
            and self.selected_group_idx < len(self.synthesis_scene.object_groups)
            and self.selected_object_in_group_idx < 0
        ):
            selected_obj = self.synthesis_scene.object_groups[self.selected_group_idx]

        if selected_obj:
            imgui.text("Transform Selected:")

            # Position sliders + numeric inputs
            changed_x, self.transform_position[0] = imgui.slider_float(
                "X##pos", self.transform_position[0], -10.0, 10.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##pos_x_input",
                self.transform_position[0],
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.transform_position[0] = float(inp)
                except Exception:
                    pass

            if changed_x or changed_input:
                selected_obj.position[0] = self.transform_position[0]
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

            changed_y, self.transform_position[1] = imgui.slider_float(
                "Y##pos", self.transform_position[1], -10.0, 10.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##pos_y_input",
                self.transform_position[1],
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.transform_position[1] = float(inp)
                except Exception:
                    pass

            if changed_y or changed_input:
                selected_obj.position[1] = self.transform_position[1]
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

            changed_z, self.transform_position[2] = imgui.slider_float(
                "Z##pos", self.transform_position[2], -10.0, 10.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##pos_z_input",
                self.transform_position[2],
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.transform_position[2] = float(inp)
                except Exception:
                    pass

            if changed_z or changed_input:
                selected_obj.position[2] = self.transform_position[2]
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

            imgui.spacing()

            # Rotation sliders
            imgui.text("Rotation (degrees):")
            changed_rx, self.transform_rotation[0] = imgui.slider_float(
                "X##rot", self.transform_rotation[0], -180.0, 180.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##rot_x_input",
                self.transform_rotation[0],
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.transform_rotation[0] = float(inp)
                except Exception:
                    pass

            if changed_rx or changed_input:
                selected_obj.rotation[0] = self.transform_rotation[0]
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

            changed_ry, self.transform_rotation[1] = imgui.slider_float(
                "Y##rot", self.transform_rotation[1], -180.0, 180.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##rot_y_input",
                self.transform_rotation[1],
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.transform_rotation[1] = float(inp)
                except Exception:
                    pass

            if changed_ry or changed_input:
                selected_obj.rotation[1] = self.transform_rotation[1]
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

            changed_rz, self.transform_rotation[2] = imgui.slider_float(
                "Z##rot", self.transform_rotation[2], -180.0, 180.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##rot_z_input",
                self.transform_rotation[2],
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.transform_rotation[2] = float(inp)
                except Exception:
                    pass

            if changed_rz or changed_input:
                selected_obj.rotation[2] = self.transform_rotation[2]
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())

            imgui.spacing()

            # Scale slider
            imgui.text("Scale:")
            changed_scale, self.transform_scale = imgui.slider_float(
                "Scale", self.transform_scale, 0.0, 100.0
            )
            imgui.same_line()
            changed_input, input_val = imgui.input_float(
                "##scale_input",
                self.transform_scale,
                step=0.0,
                step_fast=0.0,
                format="%.8g",
            )
            if changed_input:
                try:
                    v = float(input_val)
                    if v < 0.0:
                        v = 0.0
                    self.transform_scale = v
                except Exception:
                    pass

            if changed_scale or changed_input:
                selected_obj.scale_factor = self.transform_scale
                self.synthesis_scene._rebuild_scene()
                self.renderer.set_scene(self.synthesis_scene.get_root())
        else:
            imgui.text("Select an object or group to transform")

        imgui.separator()

        # Generation section
        if imgui.collapsing_header("Data Generation", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text(f"Output Dir: {self.output_dir}")

            if imgui.button("Browse Output Directory"):
                dir_path = filedialog.askdirectory(
                    title="Select Output Directory",
                    initialdir=self.output_dir if self.output_dir else ".",
                )
                if dir_path:
                    self.output_dir = dir_path
                    print(f"Output directory set to: {self.output_dir}")

            if imgui.button("Generate Training Data"):
                if len(self.candidate_objects) == 0:
                    print(
                        "Warning: No candidate objects! Add objects to candidates list first."
                    )
                else:
                    self._generate_data()

            imgui.spacing()
            imgui.text(f"Images generated: {self.image_counter}")
            imgui.text(f"Candidate objects: {len(self.candidate_objects)}")

            # Animation options
            if len(self.recorded_animations) > 0:
                imgui.spacing()
                imgui.separator()
                imgui.text("Animation Options:")

                # Animation selection for synthesis
                imgui.spacing()
                imgui.text("Select Animation:")

                if len(self.recorded_animations) == 0:
                    imgui.text_colored(
                        "No animations recorded yet",
                        0.7,
                        0.7,
                        0.0,
                        1.0,
                    )
                else:
                    for idx, anim in enumerate(self.recorded_animations):
                        label = f"{anim['name']} ({len(anim['keypoints'])} keypoints)"
                        if imgui.selectable(label, self.selected_animation_idx == idx)[
                            0
                        ]:
                            if self.selected_animation_idx == idx:
                                self.selected_animation_idx = -1
                            else:
                                self.selected_animation_idx = idx

                imgui.spacing()
                if self.selected_animation_idx >= 0:
                    anim = self.recorded_animations[self.selected_animation_idx]
                    imgui.text(f"Using animation: {anim['name']}")
                    imgui.text(f"Keypoints: {len(anim['keypoints'])}")

                    changed_slider, self.frames_per_capture = imgui.slider_int(
                        "Frames per Segment", self.frames_per_capture, 1, 100
                    )
                    imgui.same_line()
                    changed_input, inp_int = imgui.input_int(
                        "##frames_input", self.frames_per_capture
                    )
                    if changed_input:
                        try:
                            self.frames_per_capture = int(inp_int)
                        except Exception:
                            pass
                    if changed_slider or changed_input:
                        # keep within bounds
                        self.frames_per_capture = max(
                            1, min(100, int(self.frames_per_capture))
                        )

                    total_frames = (
                        len(anim["keypoints"]) - 1
                    ) * self.frames_per_capture
                    imgui.text(f"Total frames to generate: {total_frames}")
                else:
                    imgui.text_colored(
                        "No animation selected",
                        0.7,
                        0.7,
                        0.0,
                        1.0,
                    )

        imgui.separator()

        # Camera info
        if imgui.collapsing_header("Camera Controls")[0]:
            imgui.text("Control Mode:")

            # Only allow mode switch when no object is selected
            if self.selected_object_idx >= 0:
                imgui.text_colored(
                    "(Disabled while editing object)", 0.7, 0.7, 0.0, 1.0
                )

            # Toggle FPS camera mode
            changed, self.use_camera_mode = imgui.checkbox(
                "Enable FPS Camera (Q to toggle)", self.use_camera_mode
            )
            if changed and self.selected_object_idx < 0:
                if self.use_camera_mode:
                    self._enable_camera_mode()
                else:
                    self._disable_camera_mode()

            imgui.spacing()

            if self.use_camera_mode:
                imgui.text("FPS Camera Controls:")
                imgui.text("WASD - Move camera")
                imgui.text("Mouse move - Look around")
                imgui.text("Scroll - Zoom in/out")
                imgui.text("Q - Disable FPS camera")
                imgui.text("R - Record keypoint (animation)")
                cam_pos = self.renderer.camera.position
                imgui.text(
                    f"Position: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})"
                )
            else:
                imgui.text("Camera Disabled:")
                imgui.text("Q - Enable FPS camera")
                imgui.text("Mouse is free for UI interaction")

            # Camera settings always visible
            imgui.spacing()
            imgui.text("Camera Settings:")
            changed_slider, self.camera_move_speed = imgui.slider_float(
                "Move Speed", self.camera_move_speed, 0.5, 20.0
            )
            imgui.same_line()
            changed_input, inp = imgui.input_float(
                "##move_speed_input",
                self.camera_move_speed,
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input:
                try:
                    self.camera_move_speed = float(inp)
                except Exception:
                    pass
            if changed_slider or changed_input:
                self.camera_move_speed = max(
                    0.1, min(100.0, float(self.camera_move_speed))
                )

            changed_slider2, self.camera_look_sensitivity = imgui.slider_float(
                "Look Sensitivity", self.camera_look_sensitivity, 0.01, 1.0
            )
            imgui.same_line()
            changed_input2, inp2 = imgui.input_float(
                "##look_sens_input",
                self.camera_look_sensitivity,
                step=0.0,
                step_fast=0.0,
                format="%.6g",
            )
            if changed_input2:
                try:
                    self.camera_look_sensitivity = float(inp2)
                except Exception:
                    pass
            if changed_slider2 or changed_input2:
                self.camera_look_sensitivity = max(
                    0.001, min(10.0, float(self.camera_look_sensitivity))
                )

            imgui.spacing()
            imgui.separator()
            imgui.text("Object Editing (when selected):")
            imgui.text("WASD - Move on horizontal plane")
            imgui.text("Shift+W/S - Move vertically")
            imgui.text("Left drag - Rotate object")
            imgui.text("Scroll - Scale object")

        imgui.separator()

        # Animation Recording
        if imgui.collapsing_header("Camera Animation")[0]:
            if not self.use_camera_mode:
                imgui.text_colored(
                    "(Switch to Camera mode to record)", 0.7, 0.7, 0.0, 1.0
                )

            imgui.text("Controls (in FPS mode):")
            imgui.text("  N - Start new animation")
            imgui.text("  R - Record keypoint")
            imgui.text("  F - Finish & save")
            imgui.text("  ESC - Cancel recording")
            imgui.spacing()

            if self.is_recording_animation:
                imgui.text_colored(
                    f"Recording... ({len(self.current_animation_keypoints)} keypoints)",
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                )
                imgui.text("Press F to finish, ESC to cancel")
            else:
                if self.use_camera_mode:
                    imgui.text("Press N to start new animation")

            imgui.spacing()
            _, self.animation_name = imgui.input_text(
                "Animation Name", self.animation_name, 64
            )

            imgui.spacing()
            imgui.text(f"Recorded Animations: {len(self.recorded_animations)}")
            imgui.text("(Select animation in 'Synthesized Dataset' section)")

        imgui.separator()

        # Lighting controls
        if imgui.collapsing_header("Lighting & Shading", imgui.TREE_NODE_DEFAULT_OPEN)[
            0
        ]:
            imgui.text("Shading Model:")

            if imgui.radio_button("Normal", self.current_shading == 0):
                self.current_shading = 0
                self.renderer.set_shading_model(ShadingModel.NORMAL)

            if imgui.radio_button("Phong", self.current_shading == 1):
                self.current_shading = 1
                self.renderer.set_shading_model(ShadingModel.PHONG)

            if imgui.radio_button("Gouraud", self.current_shading == 2):
                self.current_shading = 2
                self.renderer.set_shading_model(ShadingModel.GOURAUD)

        imgui.end()

        # Objects panel (right side - shows object list)
        try:
            imgui.set_next_window_position(self.width - 320, 10)
            imgui.set_next_window_size(300, 400)
            if imgui.begin("Objects", True)[0]:
                imgui.text("Scene Objects")
                imgui.separator()

                # Standalone Objects
                if len(self.synthesis_scene.objects) > 0:
                    imgui.text(f"Objects ({len(self.synthesis_scene.objects)}):")
                    for idx, obj in enumerate(self.synthesis_scene.objects):
                        is_selected = self.selected_object_idx == idx

                        # Selectable object
                        if imgui.selectable(obj.name, is_selected)[0]:
                            if self.selected_object_idx == idx:
                                # Deselect
                                self.selected_object_idx = -1
                            else:
                                # Select this object
                                self.selected_object_idx = idx
                                self.selected_group_idx = -1
                                self.selected_object_in_group_idx = -1
                                self.transform_position = list(obj.position)
                                self.transform_rotation = list(obj.rotation)
                                self.transform_scale = obj.scale_factor
                            self._update_object_highlights()

                        # Right-click menu
                        if imgui.is_item_clicked(1):
                            imgui.open_popup(f"obj_ctx_{idx}")
                        if imgui.begin_popup(f"obj_ctx_{idx}"):
                            if imgui.menu_item("Add to Candidates")[0]:
                                self.candidate_objects.append(
                                    {"name": obj.name, "object": obj}
                                )
                                print(f"Added to candidates: {obj.name}")
                            if imgui.menu_item("Delete")[0]:
                                self.synthesis_scene.objects.pop(idx)
                                if self.selected_object_idx == idx:
                                    self.selected_object_idx = -1
                                self.synthesis_scene._rebuild_scene()
                                self.renderer.set_scene(self.synthesis_scene.get_root())
                            imgui.end_popup()

                # Object Groups
                if len(self.synthesis_scene.object_groups) > 0:
                    imgui.spacing()
                    imgui.text(f"Groups ({len(self.synthesis_scene.object_groups)}):")
                    for grp_idx, group in enumerate(self.synthesis_scene.object_groups):
                        is_group_selected = (
                            self.selected_group_idx == grp_idx
                            and self.selected_object_in_group_idx < 0
                        )

                        # Group header
                        if imgui.selectable(f"[G] {group.name}", is_group_selected)[0]:
                            if (
                                self.selected_group_idx == grp_idx
                                and self.selected_object_in_group_idx < 0
                            ):
                                # Deselect group
                                self.selected_group_idx = -1
                            else:
                                # Select group
                                self.selected_group_idx = grp_idx
                                self.selected_object_in_group_idx = -1
                                self.selected_object_idx = -1
                                self.transform_position = list(group.position)
                                self.transform_rotation = list(group.rotation)
                                self.transform_scale = group.scale_factor
                            self._update_object_highlights()

                        # Right-click menu for group
                        if imgui.is_item_clicked(1):
                            imgui.open_popup(f"grp_ctx_{grp_idx}")
                        if imgui.begin_popup(f"grp_ctx_{grp_idx}"):
                            if imgui.menu_item("Add Group to Candidates")[0]:
                                # Add all objects in group to candidates
                                for obj in group.objects:
                                    # Check if already in candidates
                                    already_added = any(
                                        c["object"] is obj
                                        for c in self.candidate_objects
                                    )
                                    if not already_added:
                                        self.candidate_objects.append(
                                            {"name": obj.name, "object": obj}
                                        )
                                        print(f"Added to candidates: {obj.name}")
                                print(
                                    f"Added group '{group.name}' ({len(group.objects)} objects) to candidates"
                                )
                            if imgui.menu_item("Delete Group")[0]:
                                self.synthesis_scene.object_groups.pop(grp_idx)
                                if self.selected_group_idx == grp_idx:
                                    self.selected_group_idx = -1
                                    self.selected_object_in_group_idx = -1
                                self.synthesis_scene._rebuild_scene()
                                self.renderer.set_scene(self.synthesis_scene.get_root())
                            imgui.end_popup()

                        # Show group's objects (indented)
                        imgui.indent()
                        for obj_in_grp_idx, obj in enumerate(group.objects):
                            is_obj_selected = (
                                self.selected_group_idx == grp_idx
                                and self.selected_object_in_group_idx == obj_in_grp_idx
                            )

                            if imgui.selectable(f"  {obj.name}", is_obj_selected)[0]:
                                if (
                                    self.selected_group_idx == grp_idx
                                    and self.selected_object_in_group_idx
                                    == obj_in_grp_idx
                                ):
                                    # Deselect
                                    self.selected_object_in_group_idx = -1
                                else:
                                    # Select this object in group
                                    self.selected_group_idx = grp_idx
                                    self.selected_object_in_group_idx = obj_in_grp_idx
                                    self.selected_object_idx = -1
                                    self.transform_position = list(obj.position)
                                    self.transform_rotation = list(obj.rotation)
                                    self.transform_scale = obj.scale_factor
                                self._update_object_highlights()

                            # Right-click menu for object in group
                            if imgui.is_item_clicked(1):
                                imgui.open_popup(
                                    f"grp{grp_idx}_obj_ctx_{obj_in_grp_idx}"
                                )
                            if imgui.begin_popup(
                                f"grp{grp_idx}_obj_ctx_{obj_in_grp_idx}"
                            ):
                                if imgui.menu_item("Add to Candidates")[0]:
                                    self.candidate_objects.append(
                                        {"name": obj.name, "object": obj}
                                    )
                                    print(f"Added to candidates: {obj.name}")
                                imgui.end_popup()
                        imgui.unindent()

                # Clear All button
                imgui.spacing()
                imgui.separator()
                if imgui.button("Clear All Objects"):
                    self.synthesis_scene.objects.clear()
                    self.synthesis_scene.object_groups.clear()
                    self.selected_object_idx = -1
                    self.selected_group_idx = -1
                    self.selected_object_in_group_idx = -1
                    self.candidate_objects.clear()
                    self.synthesis_scene._rebuild_scene()
                    self.renderer.set_scene(self.synthesis_scene.get_root())

            imgui.end()
        except Exception as e:
            # Ensure ImGui window stack is balanced even on errors
            try:
                imgui.end()
            except Exception:
                pass
            print(f"Objects panel error: {e}")
            pass

        # Candidate Objects panel (top-right)
        try:
            imgui.set_next_window_position(self.width - 320, 420)
            imgui.set_next_window_size(300, 250)
            if imgui.begin("Candidates", True)[0]:
                imgui.text("Candidate Objects")
                imgui.separator()
                # List candidates
                remove_idx = None
                for i, c in enumerate(self.candidate_objects):
                    # Display name; allow selection
                    imgui.selectable(c["name"], False)
                    if imgui.is_item_clicked(1):
                        imgui.open_popup(f"cand_ctx_{i}")
                    if imgui.begin_popup(f"cand_ctx_{i}"):
                        if imgui.menu_item("Remove")[0]:
                            remove_idx = i
                        imgui.end_popup()

                if remove_idx is not None:
                    removed = self.candidate_objects.pop(remove_idx)
                    print(f"Removed from candidates: {removed['name']}")

            imgui.end()
        except Exception as e:
            # Ensure ImGui window stack is balanced even on errors
            try:
                imgui.end()
            except Exception:
                pass
            # Keep UI robust if ImGui features differ
            print(f"Candidates panel error: {e}")
            pass

        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def _enable_camera_mode(self):
        """Enable FPS camera mode and lock the mouse cursor"""
        if self.use_camera_mode:
            return

        self.use_camera_mode = True
        self.renderer.use_trackball = False

        # Lock cursor for FPS mode
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        print("FPS Camera enabled - Mouse locked (Press Q to disable)")

    def _disable_camera_mode(self):
        """Disable FPS camera mode and show the mouse cursor"""
        if not self.use_camera_mode:
            return

        # Restore normal cursor
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        self.use_camera_mode = False
        self.renderer.use_trackball = False

        print("FPS Camera disabled - Mouse unlocked")

    def _start_new_animation(self):
        """Start a new animation recording"""
        if not self.use_camera_mode or self.is_recording_animation:
            return

        self.is_recording_animation = True
        self.current_animation_keypoints = []
        print(f"Started recording animation: {self.animation_name}")

    def _finish_recording_animation(self):
        """Finish and save the current animation recording"""
        if not self.is_recording_animation:
            return

        if len(self.current_animation_keypoints) >= 2:
            self.recorded_animations.append(
                {
                    "name": self.animation_name,
                    "keypoints": self.current_animation_keypoints.copy(),
                }
            )
            print(
                f"Saved animation '{self.animation_name}' with {len(self.current_animation_keypoints)} keypoints"
            )
            self.animation_name = f"Animation_{len(self.recorded_animations) + 1}"
            self.is_recording_animation = False
            self.current_animation_keypoints = []
        else:
            print("Need at least 2 keypoints to save animation")

    def _cancel_recording_animation(self):
        """Cancel the current animation recording"""
        if not self.is_recording_animation:
            return

        self.is_recording_animation = False
        self.current_animation_keypoints = []
        print("Cancelled animation recording")

    def _record_keypoint(self):
        """Record a camera keypoint for animation"""
        if not self.use_camera_mode:
            return

        # Start recording if not already
        if not self.is_recording_animation:
            self.is_recording_animation = True
            self.current_animation_keypoints = []

        # Capture camera position and rotation
        cam = self.renderer.camera
        keypoint = {"position": cam.position.copy(), "yaw": cam.yaw, "pitch": cam.pitch}
        self.current_animation_keypoints.append(keypoint)
        print(
            f"Recorded keypoint {len(self.current_animation_keypoints)}: "
            f"pos=({cam.position[0]:.2f}, {cam.position[1]:.2f}, {cam.position[2]:.2f}), "
            f"yaw={cam.yaw:.2f}, pitch={cam.pitch:.2f}"
        )

    def _interpolate_camera(self, kp1, kp2, t):
        """
        Interpolate between two camera keypoints

        Args:
            kp1: First keypoint dict with position, yaw, pitch
            kp2: Second keypoint dict with position, yaw, pitch
            t: Interpolation factor (0-1)

        Returns:
            Interpolated position, yaw, pitch
        """
        import numpy as np

        # Linear interpolation for position
        pos = kp1["position"] * (1 - t) + kp2["position"] * t

        # Interpolate yaw and pitch (handle wrapping for yaw)
        yaw1, yaw2 = kp1["yaw"], kp2["yaw"]

        # Handle yaw wrapping (shortest path)
        diff = yaw2 - yaw1
        if diff > 180:
            yaw2 -= 360
        elif diff < -180:
            yaw2 += 360

        yaw = yaw1 * (1 - t) + yaw2 * t
        pitch = kp1["pitch"] * (1 - t) + kp2["pitch"] * t

        return pos, yaw, pitch

    def _set_camera_state(self, position, yaw, pitch):
        """Set camera position and orientation"""
        cam = self.renderer.camera
        cam.position = position.copy()
        cam.yaw = yaw
        cam.pitch = pitch

        # Recalculate front vector from yaw and pitch
        cp = np.cos(np.radians(cam.pitch))
        cy = np.cos(np.radians(cam.yaw))
        sp = np.sin(np.radians(cam.pitch))
        sy = np.sin(np.radians(cam.yaw))

        cam.front = np.array([cp * cy, sp, cp * sy], dtype=np.float32)
        cam._recalculate_basis()

    def _generate_data(self):
        """Generate training data: RGB, depth, segmentation, annotations"""
        # Check if we have candidate objects (they might be in groups, not standalone objects)
        if len(self.candidate_objects) == 0:
            print("No candidate objects! Add objects to candidates list first.")
            return

        print(
            f"Starting data generation with {len(self.candidate_objects)} candidate objects..."
        )

        # Check if using animation
        if self.selected_animation_idx >= 0 and self.selected_animation_idx < len(
            self.recorded_animations
        ):
            self._generate_animated_data()
            return

        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        depth_dir = output_path / "depth"
        depth_dir.mkdir(exist_ok=True)
        seg_dir = output_path / "segmentation"
        seg_dir.mkdir(exist_ok=True)
        bbox_dir = output_path / "bounding_boxes"
        bbox_dir.mkdir(exist_ok=True)

        # Image name
        image_name = f"syn_{self.image_counter:06d}"

        # 1. Capture RGB image
        rgb_data = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        rgb_array = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        rgb_array = np.flipud(rgb_array)

        # Save normal RGB image
        rgb_image = Image.fromarray(rgb_array)
        rgb_image.save(images_dir / f"{image_name}.png")

        # 2. Create and save RGB image with bounding boxes
        bbox_image = self._create_bbox_image(
            rgb_array, image_name, self.width, self.height
        )
        bbox_image.save(bbox_dir / f"{image_name}_bbox.png")

        # 3. Capture and save depth map visualization
        depth_vis = self._create_depth_visualization()
        Image.fromarray(depth_vis).save(depth_dir / f"{image_name}_depth.png")

        # 4. Create and save segmentation mask (RGB with unique class colors)
        seg_vis = self._create_segmentation_visualization()
        # Save as RGB image
        Image.fromarray(seg_vis).save(seg_dir / f"{image_name}_seg.png")

        # 5. Generate annotations
        self._generate_annotations(image_name, rgb_array.shape[1], rgb_array.shape[0])

        self.image_counter += 1
        print(f"Generated data: {image_name}")

    def _generate_animated_data(self):
        """Generate training data using camera animation"""
        animation = self.recorded_animations[self.selected_animation_idx]
        keypoints = animation["keypoints"]

        if len(keypoints) < 2:
            print("Animation needs at least 2 keypoints!")
            return

        # Save current camera state
        saved_pos = self.renderer.camera.position.copy()
        saved_yaw = self.renderer.camera.yaw
        saved_pitch = self.renderer.camera.pitch

        print(f"Starting animated synthesis with {len(keypoints)} keypoints...")
        print(f"Capturing every {self.frames_per_capture} frames")

        # Calculate total animation segments
        num_segments = len(keypoints) - 1
        frames_per_segment = self.frames_per_capture
        total_captures = 0

        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        depth_dir = output_path / "depth"
        depth_dir.mkdir(exist_ok=True)
        seg_dir = output_path / "segmentation"
        seg_dir.mkdir(exist_ok=True)
        bbox_dir = output_path / "bounding_boxes"
        bbox_dir.mkdir(exist_ok=True)

        # Iterate through animation segments
        for seg_idx in range(num_segments):
            kp1 = keypoints[seg_idx]
            kp2 = keypoints[seg_idx + 1]

            # Interpolate and capture at intervals
            for frame in range(frames_per_segment):
                t = frame / frames_per_segment
                pos, yaw, pitch = self._interpolate_camera(kp1, kp2, t)
                self._set_camera_state(pos, yaw, pitch)

                # Clear and render the scene with new camera position
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self.renderer.render(0.0)

                # Capture frame
                image_name = f"syn_{self.image_counter:06d}"

                # 1. Capture RGB image
                rgb_data = GL.glReadPixels(
                    0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
                )
                rgb_array = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                )
                rgb_array = np.flipud(rgb_array)

                # Save normal RGB image
                rgb_image = Image.fromarray(rgb_array)
                rgb_image.save(images_dir / f"{image_name}.png")

                # 2. Create and save RGB image with bounding boxes
                bbox_image = self._create_bbox_image(
                    rgb_array, image_name, self.width, self.height
                )
                bbox_image.save(bbox_dir / f"{image_name}_bbox.png")

                # 3. Capture and save depth map visualization
                depth_vis = self._create_depth_visualization()
                Image.fromarray(depth_vis).save(depth_dir / f"{image_name}_depth.png")

                # 4. Create and save segmentation mask (RGB with unique class colors)
                seg_vis = self._create_segmentation_visualization()
                # Save as RGB image
                Image.fromarray(seg_vis).save(seg_dir / f"{image_name}_seg.png")

                # 5. Generate annotations
                self._generate_annotations(
                    image_name, rgb_array.shape[1], rgb_array.shape[0]
                )

                self.image_counter += 1
                total_captures += 1

                # Update display
                glfw.poll_events()
                glfw.swap_buffers(self.window)

        # Restore camera state
        self._set_camera_state(saved_pos, saved_yaw, saved_pitch)

        print(f"Animated synthesis complete! Generated {total_captures} frames")

    def _create_bbox_image(self, rgb_array, image_name, width, height):
        """Create RGB image with bounding boxes and class labels drawn on it"""
        from PIL import ImageDraw, ImageFont

        # Create a copy of the RGB image
        bbox_image = Image.fromarray(rgb_array.copy())
        draw = ImageDraw.Draw(bbox_image)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Get projection matrices
        proj = self.renderer.camera.get_projection_matrix()
        view = self.renderer.camera.get_view_matrix()

        # Define colors for each class (vibrant colors for visibility)
        class_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Spring Green
            (255, 0, 128),  # Rose
        ]

        # Build name-based class mapping from candidate objects
        # Objects with same name get same class_id and color
        name_to_class_id = {}
        class_id_counter = 0

        for candidate in self.candidate_objects:
            candidate_name = candidate["name"]
            if candidate_name not in name_to_class_id:
                name_to_class_id[candidate_name] = class_id_counter
                class_id_counter += 1

        # Process only candidate objects (not all scene objects)
        for candidate in self.candidate_objects:
            obj = candidate["object"]

            if not obj.visible:
                continue

            # Use candidate name to determine class_id (same name = same class)
            candidate_name = candidate["name"]
            class_id = name_to_class_id[candidate_name]
            class_name = candidate_name
            color = class_colors[class_id % len(class_colors)]

            # Get all mesh vertices for accurate 2D bounding box
            vertices = obj.shape.get_all_vertices()

            if len(vertices) == 0:
                continue

            # Project to 2D using actual mesh vertices
            model = obj.get_transform().get_matrix()
            mvp = proj @ view @ model
            bbox_2d = compute_bbox_from_projection(vertices, mvp, width, height)

            if bbox_2d:
                x_min, y_min, x_max, y_max = bbox_2d
                # Validate bbox before drawing (sanity check)
                if x_max <= x_min or y_max <= y_min:
                    print(
                        f"Warning: Invalid bbox for {obj.name}: ({x_min}, {y_min}, {x_max}, {y_max})"
                    )
                    continue
                # Draw bounding box rectangle
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)

                # Draw class label background (top-left corner)
                label_text = class_name
                # Get text size for background
                bbox = draw.textbbox((x_min, y_min), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Draw filled rectangle for label background
                draw.rectangle(
                    [(x_min, y_min - text_height - 4), (x_min + text_width + 4, y_min)],
                    fill=color,
                )

                # Draw class name text in white
                draw.text(
                    (x_min + 2, y_min - text_height - 2),
                    label_text,
                    fill=(255, 255, 255),
                    font=font,
                )

        return bbox_image

    def _create_depth_visualization(self):
        """Create a visualization of the depth map with color mapping"""
        import cv2

        # Bind depth renderer FBO
        self.depth_renderer.bind()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Render scene to depth FBO
        self.renderer.render(0.0)

        # Capture depth from FBO
        depth_array = self.depth_renderer.capture_depth()
        self.depth_renderer.unbind()

        # Normalize depth to 0-255 range
        depth_normalized = (
            (depth_array - depth_array.min())
            / (depth_array.max() - depth_array.min() + 1e-6)
            * 255
        ).astype(np.uint8)

        # Return as black and white (grayscale)
        return depth_normalized

    def _create_segmentation_visualization(self):
        """Create segmentation overlay - candidate objects with unique colors over original scene"""
        # Define unique random colors for each class
        class_colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.5, 0.0),  # Orange
            (0.5, 0.0, 1.0),  # Purple
            (0.0, 1.0, 0.5),  # Spring Green
            (1.0, 0.0, 0.5),  # Rose
            (0.5, 1.0, 0.0),  # Lime
            (0.5, 0.0, 0.5),  # Dark Magenta
        ]

        # First, capture the current RGB scene (background + all objects normally rendered)
        rgb_data = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        rgb_scene = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        rgb_scene = np.flipud(rgb_scene)

        # Now render ONLY candidate objects with flat colors to segmentation FBO
        self.seg_renderer.bind()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Get projection matrices
        proj = self.renderer.camera.get_projection_matrix()
        view = self.renderer.camera.get_view_matrix()

        # Build name-based class mapping from candidate objects
        # Objects with same name get same class_id and color
        name_to_class_id = {}
        class_id_counter = 0

        for candidate in self.candidate_objects:
            candidate_name = candidate["name"]
            if candidate_name not in name_to_class_id:
                name_to_class_id[candidate_name] = class_id_counter
                class_id_counter += 1

        # Render ONLY candidate objects (not background) with class-specific colors
        for candidate in self.candidate_objects:
            obj = candidate["object"]

            if not obj.visible:
                continue

            # Use candidate name to determine class_id (same name = same class)
            candidate_name = candidate["name"]
            class_id = name_to_class_id[candidate_name]

            # Get unique color for this class
            seg_color = np.array(
                class_colors[class_id % len(class_colors)], dtype=np.float32
            )

            # Save original color
            original_color = obj.shape.base_color.copy()

            # Set segmentation color
            obj.shape.set_color(tuple(seg_color))

            # Get transform and render this object
            transform = obj.get_transform()
            model_matrix = transform.get_matrix()

            # Set transformation matrices using shape's built-in method
            obj.shape.transform(proj, view, model_matrix)

            # Disable lighting and textures for flat color rendering
            obj.shape.shader_program.activate()
            GL.glUniform1i(
                obj.shape.shading_mode_loc, 0
            )  # Set shadingMode=0 (no lighting)

            # Draw the object with textures disabled for flat color rendering
            obj.shape.draw(force_no_texture=True)

            # Restore original color
            obj.shape.set_color(tuple(original_color))

        # Capture segmentation mask (candidate objects with flat colors)
        seg_mask = self.seg_renderer.capture_segmentation()
        self.seg_renderer.unbind()

        # Create final image: overlay colored candidate objects on original scene
        # Where seg_mask is black (0,0,0) = background, use rgb_scene
        # Where seg_mask has color = candidate object, use seg_mask color
        is_background = np.all(seg_mask == 0, axis=2)
        result = rgb_scene.copy()
        result[~is_background] = seg_mask[~is_background]

        return result

    def _generate_annotations(self, image_name: str, width: int, height: int):
        """Generate COCO and YOLO annotations"""
        # Get projection matrices
        proj = self.renderer.camera.get_projection_matrix()
        view = self.renderer.camera.get_view_matrix()

        # Initialize exporters
        coco = COCOExporter(self.output_dir)
        yolo = YOLOExporter(self.output_dir)

        # Add image to COCO
        image_id = coco.add_image(f"{image_name}.png", width, height)

        # Build name-based class mapping from candidate objects
        # Objects with same name get same class_id
        name_to_class_id = {}
        class_id_counter = 0

        for candidate in self.candidate_objects:
            candidate_name = candidate["name"]
            if candidate_name not in name_to_class_id:
                name_to_class_id[candidate_name] = class_id_counter
                class_id_counter += 1

        # Process only candidate objects (not all scene objects)
        for candidate in self.candidate_objects:
            obj = candidate["object"]

            if not obj.visible:
                continue

            # Check occlusion - skip if >90% occluded
            occlusion_pct = self._compute_occlusion_percentage(obj, width, height)
            if occlusion_pct > 0.9:
                print(f"Skipping {obj.name}: {occlusion_pct*100:.1f}% occluded")
                continue

            # Use candidate name to determine class_id (same name = same class)
            candidate_name = candidate["name"]
            class_id = name_to_class_id[candidate_name]
            class_name = candidate_name

            # Add class
            coco.add_category(class_id, class_name)
            yolo.add_class(class_name)

            # Get all mesh vertices
            vertices = obj.shape.get_all_vertices()

            if len(vertices) == 0:
                continue

            # Get transform matrix
            transform = obj.get_transform()
            model_matrix = transform.get_matrix()

            # Compute MVP matrix
            mvp = proj @ view @ model_matrix

            # Project vertices to 2D and compute bounding box
            bbox_2d = compute_bbox_from_projection(vertices, mvp, width, height)

            if bbox_2d:
                # Add to COCO (using name-based class_id)
                coco.add_annotation(image_id, class_id, bbox_2d)

                # Add to YOLO (using name-based class_id)
                yolo.add_annotation(image_name, width, height, class_id, bbox_2d)

        # Save annotations (append mode)
        if self.image_counter == 0:
            coco.save()
            yolo.save_class_names()
            yolo.create_yaml()

    def _compute_occlusion_percentage(self, obj, width: int, height: int) -> float:
        """
        Compute what percentage of an object is occluded

        Returns:
            Occlusion percentage (0.0 to 1.0), where 1.0 means fully occluded
        """
        try:
            # Render scene without this object to get depth of occluders
            obj.visible = False
            self.synthesis_scene._rebuild_scene()
            self.renderer.set_scene(self.synthesis_scene.get_root())

            # Render and capture depth with object hidden
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_renderer.fbo)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            self.renderer.render(0.0)
            depth_without_obj = self.depth_renderer.capture_depth()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

            # Render scene with this object to get its depth
            obj.visible = True
            self.synthesis_scene._rebuild_scene()
            self.renderer.set_scene(self.synthesis_scene.get_root())

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_renderer.fbo)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            self.renderer.render(0.0)
            depth_with_obj = self.depth_renderer.capture_depth()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

            # Compare: pixels where depth changed are pixels belonging to our object
            # If depth_with_obj < depth_without_obj, that pixel shows our object
            epsilon = 0.001  # Depth comparison threshold
            object_pixels = np.abs(depth_with_obj - depth_without_obj) > epsilon
            total_object_pixels = np.sum(object_pixels)

            if total_object_pixels == 0:
                return 1.0  # No visible pixels = fully occluded

            # Now check how many of these object pixels are actually visible
            # A pixel is visible if depth_with_obj at that pixel is the front-most
            # This means depth_with_obj[pixel] < depth_without_obj[pixel]
            visible_pixels = (
                depth_with_obj < depth_without_obj - epsilon
            ) & object_pixels
            visible_count = np.sum(visible_pixels)

            occlusion_pct = 1.0 - (visible_count / total_object_pixels)
            return occlusion_pct

        except Exception as e:
            print(f"Error computing occlusion for {obj.name}: {e}")
            return 0.0  # Assume visible if error

    def get_aspect_ratio(self):
        """Get window aspect ratio"""
        # Protect against a zero height (can happen when window is minimized)
        try:
            h = float(self.height)
        except Exception:
            return 1.0
        if h <= 0.0:
            return 1.0
        return float(self.width) / h

    @property
    def winsize(self):
        """Get window size"""
        # Ensure we never return a zero dimension (GL viewport and trackball expect >=1)
        return (max(1, int(self.width)), max(1, int(self.height)))

    def run(self):
        """Main application loop"""
        last_time = glfw.get_time()

        while not glfw.window_should_close(self.window):
            # Calculate delta time
            current_time = glfw.get_time()
            delta_time = current_time - last_time
            last_time = current_time

            # Process input
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            # Process keyboard for camera movement or object transformation
            self._process_keyboard_input(delta_time)

            # Apply lighting to all objects
            self._apply_lighting_to_scene()

            # Clear buffers
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Render 3D scene
            self.renderer.render(delta_time)

            # Render UI
            self._render_ui()

            # Swap buffers
            glfw.swap_buffers(self.window)

        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.renderer.cleanup()
        self.depth_renderer.cleanup()
        self.seg_renderer.cleanup()
        # Grid removed: no cleanup required
        self.imgui_impl.shutdown()
        self.tk_root.destroy()
        glfw.terminate()


def main():
    """Main entry point"""
    app = DataSynthesisApp(1280, 720)
    app.run()


if __name__ == "__main__":
    main()
