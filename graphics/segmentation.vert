"""
Segmentation shader - outputs solid color per object
"""
#version 330 core

layout (location = 0) in vec3 position;

uniform mat4 transform;
uniform mat4 camera;
uniform mat4 project;

void main() {
    gl_Position = project * camera * transform * vec4(position, 1.0);
}
