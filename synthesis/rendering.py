"""
Rendering utilities for depth maps and segmentation masks
"""

import numpy as np
from OpenGL import GL
from PIL import Image
from typing import Tuple, Dict, List
import cv2


class DepthRenderer:
    """Renderer for capturing depth maps"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fbo = None
        self.depth_texture = None
        self._setup_framebuffer()

    def _setup_framebuffer(self):
        """Create framebuffer for depth rendering"""
        # Create framebuffer
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        # Create depth texture
        self.depth_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT32,
            self.width,
            self.height,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        # Attach depth texture to framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self.depth_texture,
            0,
        )

        # We don't need color attachment for depth-only rendering
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        # Check framebuffer status
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer not complete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def capture_depth(self) -> np.ndarray:
        """
        Capture current depth buffer

        Returns:
            Depth map as numpy array (height, width) with values 0-1
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        # Read depth buffer
        depth_data = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Convert to numpy array
        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        depth_array = depth_array.reshape(self.height, self.width)

        # Flip vertically (OpenGL origin is bottom-left)
        depth_array = np.flipud(depth_array)

        return depth_array

    def depth_to_image(
        self, depth_array: np.ndarray, near: float = 0.1, far: float = 100.0
    ) -> np.ndarray:
        """
        Convert depth values to grayscale image

        Args:
            depth_array: Raw depth values (0-1)
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            Grayscale image (height, width) with values 0-255
        """
        # Linearize depth values
        z_ndc = 2.0 * depth_array - 1.0
        z_linear = (2.0 * near * far) / (far + near - z_ndc * (far - near))

        # Normalize to 0-1 range
        z_normalized = (z_linear - near) / (far - near)
        z_normalized = np.clip(z_normalized, 0, 1)

        # Convert to 8-bit grayscale
        depth_image = (z_normalized * 255).astype(np.uint8)

        return depth_image

    def bind(self):
        """Bind framebuffer for rendering"""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.width, self.height)

    def unbind(self):
        """Unbind framebuffer"""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def cleanup(self):
        """Cleanup OpenGL resources"""
        if self.depth_texture:
            GL.glDeleteTextures([self.depth_texture])
        if self.fbo:
            GL.glDeleteFramebuffers(1, [self.fbo])


class SegmentationRenderer:
    """Renderer for capturing segmentation masks"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fbo = None
        self.color_texture = None
        self.depth_renderbuffer = None
        self._setup_framebuffer()

    def _setup_framebuffer(self):
        """Create framebuffer for segmentation rendering"""
        # Create framebuffer
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        # Create color texture for object IDs
        self.color_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB,
            self.width,
            self.height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

        # Attach color texture
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.color_texture,
            0,
        )

        # Create depth renderbuffer
        self.depth_renderbuffer = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_renderbuffer)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, self.width, self.height
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            self.depth_renderbuffer,
        )

        # Check framebuffer status
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer not complete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def capture_segmentation(self) -> np.ndarray:
        """
        Capture current segmentation mask

        Returns:
            Segmentation mask as RGB numpy array (height, width, 3)
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        # Read color buffer
        color_data = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Convert to numpy array
        seg_array = np.frombuffer(color_data, dtype=np.uint8)
        seg_array = seg_array.reshape(self.height, self.width, 3)

        # Flip vertically
        seg_array = np.flipud(seg_array)

        return seg_array

    def bind(self):
        """Bind framebuffer for rendering"""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background = no object

    def unbind(self):
        """Unbind framebuffer"""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def cleanup(self):
        """Cleanup OpenGL resources"""
        if self.color_texture:
            GL.glDeleteTextures([self.color_texture])
        if self.depth_renderbuffer:
            GL.glDeleteRenderbuffers(1, [self.depth_renderbuffer])
        if self.fbo:
            GL.glDeleteFramebuffers(1, [self.fbo])


def extract_object_masks(
    segmentation: np.ndarray, object_colors: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Extract individual object masks from segmentation image

    Args:
        segmentation: RGB segmentation image (H, W, 3)
        object_colors: Dict mapping object_id to RGB color

    Returns:
        Dict mapping object_id to binary mask (H, W)
    """
    masks = {}

    for obj_id, color in object_colors.items():
        # Find pixels matching this color (with small tolerance)
        color_255 = (color * 255).astype(np.uint8)
        diff = np.abs(segmentation.astype(np.int16) - color_255)
        mask = np.all(diff <= 5, axis=2).astype(np.uint8)

        masks[obj_id] = mask

    return masks


def compute_bounding_box_2d(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute 2D bounding box from binary mask

    Args:
        mask: Binary mask (H, W)

    Returns:
        (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)
