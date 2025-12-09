#version 330 core

in vec2 v_ndc;
out vec4 FragColor;

uniform mat4 projection;
uniform mat4 view;
uniform float gridHeight;
uniform float gridScale;
uniform vec4 gridColorThin;
uniform vec4 gridColorThick;
uniform vec4 xAxisColor;
uniform vec4 zAxisColor;

// Unproject NDC (x,y) at depth z (-1 near, +1 far) into world space
vec3 unproject(vec2 ndc, float z)
{
    vec4 clip = vec4(ndc, z, 1.0);
    mat4 invVP = inverse(projection * view);
    vec4 world = invVP * clip;
    return world.xyz / world.w;
}

vec4 gridColorAt(vec3 pos)
{
    vec2 coord = pos.xz * gridScale;
    vec2 fw = fwidth(coord);
    vec2 g = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2(1e-6));
    float line = min(g.x, g.y);

    vec4 color = gridColorThin;

    // Thicker lines every 10 units
    if (mod(floor(pos.x * gridScale), 10.0) == 0.0 || mod(floor(pos.z * gridScale), 10.0) == 0.0)
        color = gridColorThick;

    // Axis colors near origin
    if (abs(pos.z) < 1e-2) color = xAxisColor;
    if (abs(pos.x) < 1e-2) color = zAxisColor;

    color.a *= 1.0 - clamp(line, 0.0, 1.0);
    return color;
}

void main()
{
    // Convert vertex v_ndc (-1..1) to NDC coords
    vec2 ndc = v_ndc;

    // Unproject near and far points
    vec3 nearP = unproject(ndc, -1.0);
    vec3 farP = unproject(ndc, 1.0);

    // Intersect ray with plane y = gridHeight
    float denom = (farP.y - nearP.y);
    if (abs(denom) < 1e-6) discard;
    float t = (gridHeight - nearP.y) / denom;
    if (t < 0.0 || t > 1.0) discard;

    vec3 worldPos = mix(nearP, farP, t);

    // Depth
    vec4 clip = projection * view * vec4(worldPos, 1.0);
    float depth = clip.z / clip.w;
    gl_FragDepth = depth;

    // Compute color and fade with distance
    vec4 c = gridColorAt(worldPos);
    vec3 camPos = inverse(view)[3].xyz;
    float dist = length(worldPos - camPos);
    float fade = 1.0 - clamp(dist / 100.0, 0.0, 1.0);
    c.a *= fade;

    if (c.a < 0.01) discard;

    FragColor = c;
}
