#version 330 core

out vec4 color;

in vec3 vertexColor;
in vec3 litColor;  // Pre-calculated and interpolated lighting color
in vec2 textureCoord;

uniform sampler2D textureData;
uniform bool use_texture;
uniform int shadingMode; // 0 = normal visualization, 2 = Gouraud

void main()
{
    if (shadingMode == 0) {
        color = vec4(vertexColor, 1.0);
        return;
    }
    
    vec3 finalColor = litColor;
    
    if (use_texture) {
        vec3 texColor = texture(textureData, textureCoord).rgb;
        finalColor = mix(finalColor, texColor, 0.5); // finalColor * (1.0 - 0.5) + texColor * 0.5
    }
    
    color = vec4(finalColor, 1.0);
}
