# 
#
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 norm;
layout (location = 3) in vec2 texture;

out vec3 vertexColor;
out vec3 vertexNorm;
out vec3 vertexCoord;
out vec2 textureCoord;

uniform mat4 transform;
uniform mat4 camera;
uniform mat4 project;

void main()
{
    vertexColor = color;

    vec4 vertexCoord_homo = camera * transform * vec4(position, 1.0);
    vertexCoord = vec3(vertexCoord_homo) / vertexCoord_homo.w;

    vertexNorm = mat3(transpose(inverse(camera * transform))) * norm; // costly, inverse matrix should be computer on CPU

    textureCoord = texture;
    gl_Position = project * camera * transform * vec4(position, 1.0);
}