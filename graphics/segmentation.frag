"""
Segmentation shader - outputs solid color per object
"""
#version 330 core

out vec4 FragColor;

uniform vec3 segmentationColor;

void main() {
    FragColor = vec4(segmentationColor, 1.0);
}
