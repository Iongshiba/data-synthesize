#version 330 core

out vec4 color;

in vec3 vertexColor; // this turn into position for fragment, not vertex anymore
in vec3 vertexNorm;
in vec3 vertexCoord;
in vec2 textureCoord;

uniform sampler2D textureData;
uniform bool use_texture;

uniform mat3 I_lights;
uniform mat3 K_materials;

uniform float shininess;

uniform vec3 lightCoord;

uniform int shadingMode; // 0 = normal visualization, 1 = Phong

void main()
{
    vec3 finalColor;
    if (shadingMode == 0) {
        finalColor = vertexColor;
    }
    else
    {
        // diffuse
        vec3 vectorNorm = normalize(vertexNorm);
        vec3 lightDirection = normalize(lightCoord - vertexCoord);

        // specular
        vec3 cameraDirection = normalize(-vertexCoord);
        vec3 reflectDirection = reflect(-lightDirection, vectorNorm);

        vec3 g = vec3(
            max(dot(lightDirection, vectorNorm), 0.0),
            pow(max(dot(cameraDirection, reflectDirection), 0.0), shininess),
            0.0
        );
        vec3 fragColor = matrixCompMult(K_materials, I_lights) * g;
        finalColor = vertexColor * 0.5 + fragColor * 0.5;
    }


    if (use_texture)
    {
        vec3 texColor = texture(textureData, textureCoord).rgb;
        finalColor = mix(finalColor, texColor, 0.8);
    }

    color = vec4(finalColor, 1.0);
}