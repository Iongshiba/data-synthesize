import numpy as np


class Vertex:
    def __init__(self, x, y, z, r=None, g=None, b=None):
        self.vertex = np.array([x, y, z], dtype=np.float32)
        r = np.random.uniform(0.5, 0.7) if r is None else r  # small red
        g = np.random.uniform(0.5, 0.7) if g is None else g  # small green
        b = np.random.uniform(0.5, 0.8) if b is None else b  # strong blue

        self.color = np.array([r, g, b], dtype=np.float32)
