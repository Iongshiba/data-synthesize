# Data Synthesis for Computer Vision

A Python-based OpenGL application for generating synthetic training data with automatic annotations for object detection and segmentation tasks. Load 3D models (OBJ, GLTF, GLB, FBX), place them in scenes, and export RGB images with depth maps, segmentation masks, and COCO/YOLO annotations.

![Data Synthesis Demo](https://via.placeholder.com/800x400?text=Data+Synthesis+Application)

## Features

- **3D Model Loading**: Support for OBJ, GLTF, GLB, FBX, PLY formats with automatic texture mapping
- **Scene Composition**: Interactive placement and transformation of 3D objects
- **Camera Animation**: Record camera keypoints for smooth animated capture sequences
- **Multi-format Export**:
  - RGB images
  - Depth maps (grayscale, near=white, far=black)
  - Segmentation masks (per-object colored overlays)
  - Bounding boxes (visualized on images)
  - COCO format annotations (with polygon segmentation)
  - YOLO format annotations
- **Material-based Labeling**: Each mesh/material gets its own class label automatically
- **Real-time Shading**: Phong, Gouraud, and flat shading with dynamic lighting

## Requirements

- Python 3.8+
- OpenGL-capable graphics card
- Windows/Linux/macOS

## Setup

### 1. Install Python

Make sure you have Python 3.8 or higher installed:

```bash
python --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/Iongshiba/data-synthesize.git
cd data-synthesize
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `numpy` - Numerical computations
- `scipy` - Scientific computing
- `pillow` - Image processing
- `opencv-python` - Computer vision (for segmentation masks)
- `PyOpenGL` - OpenGL bindings
- `glfw` - Window and input management
- `imgui[glfw]` - UI rendering
- `pyassimp` - 3D model loading (GLTF, FBX, etc.)
- `plyfile` - PLY format support

### 4. Run the Application

```bash
python data_synthesis_app.py
```

The application window will open with a 3D viewport and UI panels.

## How to Use

### Loading 3D Assets

1. **Prepare your assets** in this structure:
   ```
   assets/
   └── your_object/
       ├── source/
       │   └── model.obj (or .gltf, .glb, .fbx)
       └── textures/ (optional)
           └── texture.png
   ```

2. **Click "Browse Assets"** in the UI
3. **Select the asset folder** (e.g., `assets/your_object/`)
4. The model will be loaded as an object group with individual meshes

### Scene Composition

- **Object List Panel**: Shows all loaded objects and groups
- **Select objects** by clicking them in the list
- **Transform selected objects**:
  - **Position**: Use X/Y/Z sliders or input fields
  - **Rotation**: Use X/Y/Z rotation sliders (degrees)
  - **Scale**: Use scale slider or scroll wheel
- **Add to Candidates**: Right-click objects → "Add to Candidates" (required for annotation generation)

### Camera Controls

**Toggle FPS Camera** (press `Q` or checkbox):
- **WASD** - Move camera forward/left/backward/right
- **Mouse move** - Look around
- **Scroll** - Zoom in/out

**Camera Animation** (FPS mode only):
- **N** - Start new animation recording
- **R** - Record current camera position/orientation as keypoint
- **F** - Finish and save animation
- **ESC** - Cancel recording

### Generating Training Data

1. **Add objects to candidates** (right-click → "Add to Candidates")
2. **Configure output**:
   - Set output directory path
   - (Optional) Select a recorded animation for multi-frame capture
   - (Optional) Set "Frames per Segment" for animation interpolation
3. **Click "Generate Data"**

**Output structure:**
```
output/
├── images/               # RGB images
│   └── syn_000000.png
├── depth/                # Depth visualizations
│   └── syn_000000_depth.png
├── segmentation/         # Segmentation overlays
│   └── syn_000000_seg.png
├── bounding_boxes/       # RGB with bbox visualization
│   └── syn_000000_bbox.png
├── labels/               # YOLO annotations (per-image)
│   └── syn_000000.txt
├── annotations.json      # COCO format (all images)
├── classes.txt           # Class names (YOLO)
└── data.yaml             # YOLO dataset config
```

### Shading and Lighting

- **I** - Normal shading (shows surface normals)
- **O** - Gouraud shading (per-vertex lighting)
- **P** - Phong shading (per-pixel lighting, default)
- **U** - Randomize light position and intensity
- Adjust light properties via UI sliders

## Controls Reference

| Key/Action | Function |
|------------|----------|
| **Q** | Toggle FPS camera mode |
| **WASD** | Move camera (FPS mode) |
| **Mouse** | Look around (FPS mode) |
| **Scroll** | Zoom / Scale selected object |
| **N** | Start animation recording (FPS mode) |
| **R** | Record camera keypoint (FPS mode) |
| **F** | Finish animation recording (FPS mode) |
| **ESC** | Cancel animation / Quit |
| **I/O/P** | Switch shading model |
| **U** | Randomize lighting |

## Advanced Features

### Material-Based Class Labeling

When loading OBJ files with MTL materials, each material/mesh becomes a separate class:
- **OBJ mesh name** (from MTL `usemtl` directive) → **Class name**
- Example: `car.obj` with materials "body", "wheel", "window" → 3 classes

This enables training models to detect individual parts of complex objects.

### Camera Animation

Record smooth camera paths for diverse viewpoints:
1. Enable FPS camera (`Q`)
2. Press `N` to start recording
3. Move to desired positions and press `R` at each keypoint
4. Press `F` to finish (requires ≥2 keypoints)
5. Select the animation in "Animation Recording" panel
6. Set "Frames per Segment" (interpolation density)
7. Click "Generate Data" to capture all interpolated frames

### Depth Map Inversion

Depth maps are exported with **near pixels = white**, **far pixels = black** (inverted Z-buffer) for better visualization and compatibility with common depth estimation tasks.

## Project Structure

```
data-synthesize/
├── data_synthesis_app.py    # Main application entry point
├── requirements.txt          # Python dependencies
├── config/                   # Configuration and enums
│   ├── __init__.py
│   ├── enums.py              # ShadingModel enum
│   └── palette.py            # Color palettes
├── graphics/                 # OpenGL rendering primitives
│   ├── buffer.py             # VAO/VBO/EBO wrappers
│   ├── shader.py             # Shader compilation
│   ├── texture.py            # Texture loading
│   ├── scene.py              # Scene graph nodes
│   └── *.vert, *.frag        # GLSL shaders
├── rendering/                # High-level rendering
│   ├── camera.py             # Camera (FPS and trackball)
│   ├── renderer.py           # Main renderer
│   ├── world.py              # Transform utilities
│   └── animation.py          # Animation system
├── shape/                    # 3D shape abstractions
│   ├── base.py               # Shape base class
│   └── mesh_loader.py        # OBJ/GLTF/PLY loaders
├── synthesis/                # Data generation pipeline
│   ├── scene.py              # Synthesis scene management
│   ├── rendering.py          # Depth/segmentation renderers
│   ├── export.py             # COCO/YOLO exporters
│   └── placement.py          # Object placement utilities
├── utils/                    # Utility functions
│   ├── misc.py               # Model loading, texture utils
│   └── transform.py          # Matrix transformations
└── assets/                   # 3D models and textures
    └── your_model/
        ├── source/
        │   └── model.obj
        └── textures/
            └── texture.png
```

## Development

### Adding New 3D Model Formats

Extend `utils/misc.py::load_model()` to support additional formats via `pyassimp` or custom parsers.

### Custom Shaders

Add new shader pairs (`.vert` + `.frag`) in `graphics/` and register them in the shape initialization.

### Extending Export Formats

Modify `synthesis/export.py` to add new annotation formats (e.g., Pascal VOC, TFRecord).

### Architecture Overview

- **Scene Graph**: Hierarchical node structure (`graphics/scene.py`)
- **Rendering Pipeline**: Camera → Scene → Shaders → Framebuffer
- **Data Generation**: Scene → Depth FBO → Segmentation FBO → Annotations
- **UI**: ImGui integration for interactive controls

## Troubleshooting

### OpenGL Context Errors

**Linux users**: If you encounter display issues, set the OpenGL platform:
```bash
export PYOPENGL_PLATFORM=egl
```

**WSL users**: Install Mesa OpenGL:
```bash
sudo apt install libgl1-mesa-glx
```

### Model Loading Issues

- **Textures not loading**: Ensure texture files are in `textures/` folder next to the model
- **FBX/GLTF errors**: Install the latest `pyassimp` and ensure `assimp` DLL/library is available
- **Large models**: May take time to load; check terminal for progress messages

### Missing Annotations

- **COCO empty**: Ensure objects are added to candidates (right-click → "Add to Candidates")
- **YOLO missing labels**: Check `output/labels/` directory was created
- **Segmentation masks black**: Verify objects are visible and have valid geometry

### Performance Issues

- **Reduce model complexity**: Decimate meshes before importing
- **Lower resolution**: Objects with high poly counts may slow rendering
- **Disable real-time updates**: Minimize UI interactions during batch generation