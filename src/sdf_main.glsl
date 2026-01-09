
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return (cos(angle) + 1.0) / 2.0;
    }

    return 1.0;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

// exponential
float smin( float a, float b, float k )
{
    k *= 1.0;
    float r = exp2(-a/k) + exp2(-b/k);
    return -k*log2(r);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    d = smin(d, sdSphere((p - vec3(0.0, 0.55, -0.7)), 0.27), 0.05);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);

    vec4 white = vec4(sdSphere((p - vec3(0.0, 0.5, -0.4)), 0.15), 1.0, 1.0, 1.0);
    if (white.x < res.x) {
        res = white;
    }

    vec4 blue = vec4(sdSphere((p - vec3(0.0, 0.5, -0.2)), 0.003), 0.0, 0.0, 1.0);
    if (blue.x < res.x) {
        res = blue;
    }

    vec4 black = vec4(sdSphere((p - vec3(0.0, 0.5, -0.1)), 0.001), 0.0, 0.0, 0.0);
    if (black.x < res.x) {
        res = black;
    }

    return vec4(smin(black.x, smin(white.x, blue.x, 0.05), 0.05), res.y, res.z, res.w);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.15, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    float leftHandCos = lazycos(5.0 * iTime);
    float leftHandSin = sqrt(1.0 - leftHandCos * leftHandCos);
    mat2 leftHandRot = mat2(leftHandCos, -leftHandSin, leftHandSin, leftHandCos);

    vec3 leftHandShoulder = vec3(-0.35, 0.35, -0.7);
    vec2 leftHandDelta = leftHandRot * vec2(0.1, 0.173);
    vec3 leftHandEnd = leftHandShoulder;
    leftHandEnd.xy -= leftHandDelta;

    vec4 leftHand = vec4(sdCapsule(p, leftHandShoulder, leftHandEnd, 0.03), 0.0, 1.0, 0.0);
    if (leftHand.x < res.x) {
        res = leftHand;
    }

    vec4 rightHand = vec4(sdCapsule(p, vec3(0.35, 0.35, -0.7), vec3(0.45, 0.35 - 0.173, -0.7), 0.03), 0.0, 1.0, 0.0);
    if (rightHand.x < res.x) {
        res = rightHand;
    }

    vec4 leftFoot = vec4(sdCapsule(p, vec3(-0.1, -0.1, -0.7), vec3(-0.1, 0.05, -0.7), 0.045), 0.0, 1.0, 0.0);
    if (leftFoot.x < res.x) {
        res = leftFoot;
    }

    vec4 rightFoot = vec4(sdCapsule(p, vec3(0.1, -0.1, -0.7), vec3(0.1, 0.05, -0.7), 0.045), 0.0, 1.0, 0.0);
    if (rightFoot.x < res.x) {
        res = rightFoot;
    }

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
    sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
    sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{

    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);


    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));


    vec4 res = raycast(ray_origin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}