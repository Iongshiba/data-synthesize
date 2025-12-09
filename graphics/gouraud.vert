#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 norm;
layout (location = 3) in vec2 texture;

out vec3 vertexColor;
out vec3 litColor;  // Pre-calculated lighting color from vertex shader
out vec2 textureCoord;

uniform mat4 transform;
uniform mat4 camera;
uniform mat4 project;

uniform mat3 I_lights;
uniform mat3 K_materials;
uniform float shininess;
uniform vec3 lightCoord;
uniform int shadingMode; // 0 = normal visualization, 1 = Gouraud

void main()
{
    vertexColor = color;
    textureCoord = texture;
    
    // Transform vertex to eye-space
    vec4 vertexCoord_homo = camera * transform * vec4(position, 1.0);
    vec3 vertexCoord = vec3(vertexCoord_homo) / vertexCoord_homo.w;
    
    // Transform normal to eye-space
    vec3 vertexNorm = mat3(transpose(inverse(camera * transform))) * norm;
    
    if (shadingMode == 2) {
        // Normalize vectors
        vec3 N = normalize(vertexNorm);
        vec3 L = normalize(lightCoord - vertexCoord);
        vec3 V = normalize(-vertexCoord);  // Camera at origin in eye-space
        vec3 R = reflect(-L, N);
        
        // Diffuse component
        float diffuse = max(dot(L, N), 0.0);
        
        // Specular component
        float specular = pow(max(dot(V, R), 0.0), shininess);
        
        // Combine lighting components
        vec3 g = vec3(diffuse, specular, 0.0);
        vec3 lighting = matrixCompMult(K_materials, I_lights) * g;
        
        // Blend with vertex color
        litColor = color * 0.5 + lighting * 0.5;
    } else {
        // Normal visualization mode
        litColor = color;
    }
    
    gl_Position = project * camera * transform * vec4(position, 1.0);
}
