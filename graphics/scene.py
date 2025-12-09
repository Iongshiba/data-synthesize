import numpy as np
from shape.base import Shape, Part
from rendering.world import Transform


class Node:
    def __init__(
        self,
        name: str = "Node",
        children: list["Node"] = None,
    ):
        self.name: str = name
        self.children: list["Node"] = children if children else []

    def add(self, child):
        self.children.append(child)

    def draw(self, parent_matrix, view, proj):
        if parent_matrix is None:
            parent_matrix = np.identity(4)
        for child in self.children:
            child.draw(parent_matrix, view, proj)


class TransformNode(Node):
    def __init__(
        self,
        name: str = "TransformNode",
        transform: Transform = None,
        children: list["Node"] = None,
    ):
        super().__init__(name, children)
        self.transform: Transform = transform if transform else Transform()

    def draw(self, parent_matrix, view, proj):
        current = np.dot(parent_matrix, self.transform.get_matrix())
        for child in self.children:
            child.draw(current, view, proj)


class GeometryNode(Node):
    def __init__(
        self,
        name: str = "GeometryNode",
        shape: Shape = None,
    ):
        super().__init__(name)
        self.shape = shape

    def draw(self, parent_matrix, view, proj):
        self.shape.transform(proj, view, parent_matrix)
        self.shape.draw()


class LightNode(Node):
    def __init__(
        self,
        name: str = "LightNode",
        shape: Shape = None,
    ):
        super().__init__(name)
        self.shape = shape

    def draw(self, parent_matrix, view, proj):
        self.shape.transform(proj, view, parent_matrix)
        self.shape.draw()
