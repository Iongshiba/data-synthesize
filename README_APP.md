**Overview**
- **Project:** 2d-3d-render (Data Synthesis Application)
- **Purpose:** An interactive OpenGL-based tool to load 3D assets, arrange them in a synthetic scene, render RGB / depth / segmentation outputs, and export COCO / YOLO annotations for training data.

**Quick Start**
- Install required packages (see `requirements_synthesis.txt`). Typical command:

```
python -m pip install -r requirements_synthesis.txt
```

- Run the application:

```
python data_synthesis_app.py
```

**High-level Architecture**
- `data_synthesis_app.py` — Main application. UI (ImGui) + scene orchestration + input handling + generation pipeline.
- `synthesis/scene.py` — Scene graph model: `SynthesisObject`, `SynthesisObjectGroup`, scene builder and helpers.
- `shape/mesh_loader.py` & `shape/base.py` — Mesh loading (OBJ/PLY/etc.), Shape abstraction, VAO setup, material/texture handling and color handling.
- `rendering/renderer.py` — Renderer glue: camera, node traversal, shading model application, and draw call orchestration.
- `synthesis/rendering.py` — Offscreen helpers: depth and segmentation renderers used for visibility checks and mask extraction.
- `synthesis/export.py` — Exporters for COCO and YOLO formats.
- `utils/misc.py` — Model/MTL parsing and texture helpers.
- `graphics/` — Low-level helpers (VAO, Shaders, Scene node classes, optional grid plane). 

**Main Features (what they do & how they work)**

- **Load 3D assets**
  - Files: .obj (legacy), and assets organized in subfolders (textures, materials) for FBX/GLTF/GLB pipelines.
  - Where: `data_synthesis_app._load_asset_from_library()` and `synthesis.scene.add_object()`.
  - Behavior: Loads geometry via `utils.misc.load_model()`, creates `MeshShape` instances, computes bounding boxes.

- **Automatic normalization (fit-to-view)**
  - Purpose: Scale/center newly loaded objects so they reasonably fit the camera view without manual flying.
  - How: `MeshShape._compute_bounding_box()` returns axis-aligned bbox; the app computes a scale factor (using `load_normalize_size`) and applies to `SynthesisObject.scale_factor` and position to center the mesh.

- **Texture/material handling**
  - `MeshShape.apply_material_textures()` tries: MTL-referenced texture -> texture file name exact -> case-insensitive filename match -> auto-load textures from candidate directories.
  - Textures are uploaded to OpenGL and stored in `MeshShape.textures`. Drawing binds the first available texture per mesh.

- **Shading models & keyboard shortcuts**
  - Supported: Normal, Phong, Gouraud.
  - Where: `Renderer.set_shading_model()` and `Shape.set_shading_mode()` integration.
  - Keyboard shortcuts: `I` = Normal, `P` = Phong, `O` = Gouraud (handled in `_on_key()`), each prints a confirmation.

- **Randomize light (U key)**
  - Purpose: Quickly vary lighting during data synthesis.
  - How: `U` randomizes `app.light_position` and `app.light_strength`, the app applies lighting each frame with `_apply_lighting_to_scene()` which calls `shape.lighting()` using current camera position and light params.

- **Object highlighting on selection**
  - Purpose: Visual feedback when selecting objects from the Objects panel.
  - How: `MeshShape` has `set_highlight()` which brightens object color (keeps `original_color`). Selection changes call `DataSynthesisApp._update_object_highlights()` which signals selected shapes to highlight/unhighlight.

- **Depth-based occlusion detection (skip highly occluded objects)**
  - Purpose: Avoid exporting bounding boxes/segmentation for objects that are largely occluded by other geometry.
  - How: In annotation generation `_compute_occlusion_percentage()`:
    - Renders depth with the target object hidden, captures depth.
    - Renders depth with the object visible, captures depth.
    - Compares depth buffers to find pixels contributed by the object and counts how many are visible in the final depth buffer.
    - If occlusion percent > 90% the object is skipped from COCO/YOLO annotation.
  - Offscreen rendering: `synthesis.rendering.DepthRenderer` sets up an FBO with a depth texture for reads.

- **Segmentation masks**
  - `SegmentationRenderer` (in `synthesis/rendering.py`) produces per-object segmentation masks; the app can extract per-object pixel masks used for training data.

- **Exporters (COCO & YOLO)**
  - `synthesis.export.COCOExporter` and `YOLOExporter` format annotations and image entries.
  - Bounding boxes are computed via 3D->2D projection (`compute_bbox_from_projection`) using model-view-projection (MVP) matrices derived from `Renderer.camera`.

- **Camera & Animation recording**
  - Two camera modes: normal trackball & FPS camera mode (toggle `Q`).
  - While in FPS mode the user can record keypoints (N to start, R to record, F to finish) and the app can synthesize sequences by moving along recorded keypoints.

- **Object groups and hierarchical transforms**
  - `SynthesisObjectGroup` holds multiple `SynthesisObject` entries with a group-level transform; the scene builder (`SynthesisScene._rebuild_scene`) maps groups into `TransformNode` + `GeometryNode` hierarchy used by the renderer.

- **UI (ImGui)**
  - Main control panel: load assets, transform selected object, generate dataset, camera & recording controls, lighting. Implemented in `_render_ui()` in `data_synthesis_app.py`.
  - Objects panel: lists standalone objects and groups; selecting items updates selection indices and highlights the object.

**Key files and responsibilities**
- `data_synthesis_app.py` — UI, input handlers, scene management, generation orchestration, keyboard shortcuts.
- `synthesis/scene.py` — scene model, object and group types, bounding box utilities.
- `shape/mesh_loader.py` — converts loaded mesh data to VAOs, handles textures, colors, and bounding boxes.
- `shape/base.py` — shader setup, uniform locations (lighting, material parameters), draw/transform methods.
- `rendering/renderer.py` — collects node lists, applies shading and lighting, executes draw.
- `synthesis/rendering.py` — depth/segmentation offscreen renderers and helpers.
- `synthesis/export.py` — COCO and YOLO exporters.
- `graphics/` — `buffer.py`, `shader.py`, `scene.py` node classes, small utilities used by the renderer.

**Runtime controls & keyboard shortcuts**
- `Q` — Toggle FPS camera mode (when no object is selected).
- `W/A/S/D` & Shift — Move camera (FPS) or move selected object in object-edit mode.
- `N` — Start new animation recording (FPS mode).
- `R` — Record keypoint (FPS mode).
- `F` — Finish & save animation (FPS mode).
- `I` — Switch to Normal shading.
- `P` — Switch to Phong shading.
- `O` — Switch to Gouraud shading.
- `U` — Randomize light position and intensity.

**Developer notes / extension points**
- Grid: The grid plane is implemented in `graphics/grid.py` and was previously optional in the app. It can be restored or permanently removed depending on workflow.
- Occlusion cost: Depth-based occlusion checks require multiple offscreen renders per object which can be expensive for many objects. Consider sampling or an approximate method (depth test of bounding-box projections) for speed.
- Multi-material meshes: `MeshShape` splits loaded meshes into per-material parts and binds texture per mesh part when drawing.
- Camera/trackball: `rendering.camera` provides projection and view matrices used for all projections and offscreen renders; keep sync between the renderer and depth/segmentation FBOs.

**Troubleshooting**
- ImGui assertion "Mismatched Begin/End" — ensure UI blocks call `imgui.end()`; the app recently added defensive `end()` calls in exception handlers.
- Texture not found — `MeshShape.apply_material_textures()` searches several candidate directories near the mesh; place textures next to the mesh or in a `textures/` subfolder.
- Depth read errors — ensure OpenGL context is active and the FBOs are correctly bound (the app creates `DepthRenderer` after `glfw` context initialization).

**Contact / Development**
- Source is structured for easy extension; add new exporters or augmentation hooks in `data_synthesis_app.py` generation pipeline.

---
This README describes the main workflow and implementation points. If you want, I can:
- Add a short developer `README.md` into the repo root with commands and running examples.
- Generate a diagram mapping modules to responsibilities.

