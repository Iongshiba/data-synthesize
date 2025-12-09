#version 330 core

layout (location = 0) in vec3 aPos;

out vec2 v_ndc;

void main() {
    // aPos contains NDC coordinates for the fullscreen quad
    v_ndc = aPos.xy;
    gl_Position = vec4(aPos, 1.0);
}
