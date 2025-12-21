#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    const float3              orig,
    const float3              dir,
    __global const BVHNodeGPU* nodes,
    __global const uint*      leafTriIndices,
    uint                      nfaces,
    __global const float*     vertices,
    __global const uint*      faces,
    float                     tMin,
    __private float*          outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private int*            outFaceId,
    __private float*          outU, // сюда нужно записать u рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private float*          outV) // сюда нужно записать v рассчитанный в intersect_ray_triangle(..., t, u, v)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    float tIn = 0, tOut = 0, tInLeft = 0, tOutLeft = 0, tInRight = 0, tOutRight = 0, tOutBest = FLT_MAX;
    if (!intersect_ray_aabb_any(orig, dir, nodes[rootIndex].aabb, &tIn, &tOut)) {
        return false;
    }

    bool found = false;
    uint stack[STACK_SIZE];
    int lastElementIndex = 0;
    stack[lastElementIndex++] = rootIndex;


    while (lastElementIndex > 0) {
        uint nodeId = stack[--lastElementIndex];

        if (nodeId >= leafStart) {
            // leaf
            // possibly even use tIn/tOut of AABB
            float t, u, v;
            const uint faceId = leafTriIndices[nodeId - leafStart];
            const uint3 face = vload3(faceId, faces);

            const float3 v0 = vload3(face.x, vertices);
            const float3 v1 = vload3(face.y, vertices);
            const float3 v2 = vload3(face.z, vertices);
            
            bool intersects = intersect_ray_triangle(orig, dir, v0, v1, v2, tMin, FLT_MAX, false, &t, &u, &v);
            found = found || intersects;

            if (intersects && t < *outT) {
                *outT = t;
                *outU = u;
                *outV = v;
                *outFaceId = faceId;
                found = true;
            } 
        } else {
            BVHNodeGPU curr_node = nodes[nodeId];
            uint leftChildIdx = curr_node.leftChildIndex;
            uint rightChildIdx = curr_node.rightChildIndex;
            BVHNodeGPU leftChild = nodes[leftChildIdx]; 
            BVHNodeGPU rightChild = nodes[rightChildIdx];
            bool overlapL = intersect_ray_aabb_any(orig, dir, leftChild.aabb, &tInLeft, &tOutLeft);
            bool perspectiveL = true;
            if (tInLeft > *outT) {
                perspectiveL = false;
                rassert(perspectiveL, 1234);
            }

            if (overlapL && tOutLeft < tOutBest) {
                tOutBest = tOutLeft;
            }
            bool overlapR = intersect_ray_aabb_any(orig, dir, rightChild.aabb, &tInRight, &tOutRight);
            bool perspectiveR = true;
            if (tInRight > *outT) {
                perspectiveR = false;
                rassert(perspectiveR, 12345);
            }

            if (overlapR && tOutRight < tOutBest) {
                tOutBest = tOutRight;
            }

            if (overlapL && perspectiveL && overlapR && perspectiveR) {
                // get more perspective
                if (tInLeft < tInRight) {
                    stack[lastElementIndex++] = leftChildIdx;
                    stack[lastElementIndex++] = rightChildIdx;
                } else {
                    stack[lastElementIndex++] = rightChildIdx;
                    stack[lastElementIndex++] = leftChildIdx;
                }
            } else if (overlapL && perspectiveL) {
                stack[lastElementIndex++] = leftChildIdx;
            } else if (overlapR && perspectiveR) {
                stack[lastElementIndex++] = rightChildIdx;
            }
            
        }
    } 

    return found;
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    const float3              orig,
    const float3              dir,
    __global const float*     vertices,
    __global const uint*      faces,
    __global const BVHNodeGPU* nodes,
    __global const uint*      leafTriIndices,
    uint                      nfaces,
    int                       ignore_face)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    float tIn = 0, tOut = 0;
    if (!intersect_ray_aabb_any(orig, dir, nodes[rootIndex].aabb, &tIn, &tOut)) {
        return false;
    }

    bool found = false;
    uint stack[STACK_SIZE];
    int lastElementIndex = 0;
    stack[lastElementIndex++] = rootIndex;


    while (lastElementIndex > 0) {
        uint nodeId = stack[--lastElementIndex];

        if (nodeId >= leafStart) {
            // leaf
            // possibly even use tIn/tOut of AABB
            float t, u, v;
            const uint faceId = leafTriIndices[nodeId - leafStart];
            if (faceId == ignore_face) {
                continue;
            }
            const uint3 face = vload3(faceId, faces);

            const float3 v0 = vload3(face.x, vertices);
            const float3 v1 = vload3(face.y, vertices);
            const float3 v2 = vload3(face.z, vertices);
            
            bool intersects = intersect_ray_triangle_any(orig, dir, v0, v1, v2, false, &t, &u, &v);

            found = found || intersects;
            if (found) { // strong acceleration on powerplant
                return found;
            }
        } else {
            BVHNodeGPU curr_node = nodes[nodeId];
            uint leftChildIdx = curr_node.leftChildIndex;
            uint rightChildIdx = curr_node.rightChildIndex;
            BVHNodeGPU leftChild = nodes[leftChildIdx]; 
            BVHNodeGPU rightChild = nodes[rightChildIdx];
            bool overlapL = intersect_ray_aabb_any(orig, dir, leftChild.aabb, &tIn, &tOut);
            bool overlapR = intersect_ray_aabb_any(orig, dir, rightChild.aabb, &tIn, &tOut);

            if (overlapL) {
                stack[lastElementIndex++] = leftChildIdx;
            }
            if (overlapR) {
                stack[lastElementIndex++] = rightChildIdx;
            }
            
        }
    } 

    return found;
}

// Helper: build tangent basis for a given normal
static inline void make_basis(const float3 n,
                              __private float3* t,
                              __private float3* b)
{
    // pick a non-parallel vector
    float3 up = (fabs(n.z) < 0.999f)
        ? (float3)(0.0f, 0.0f, 1.0f)
        : (float3)(0.0f, 1.0f, 0.0f);

    *t = fast_normalize(cross(up, n));
    *b = cross(n, *t);
}

__kernel void ray_tracing_render_using_lbvh(
    __global const float*      vertices,
    __global const uint*       faces,
    __global const BVHNodeGPU* bvhNodes,
    __global const uint*       leafTriIndices,
    __global int*              framebuffer_face_id,
    __global float*            framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
    uint                       nfaces)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    rassert(camera.magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 786435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin;
    float3 ray_direction;
    make_primary_ray(camera,
                     (float)i + 0.5f,
                     (float)j + 0.5f,
                     &ray_origin,
                     &ray_direction);

    float tMin      = 1e-6f;
    float tBest     = FLT_MAX;
    float uBest     = 0.0f;
    float vBest     = 0.0f;
    int   faceIdBest = -1;

    // Use BVH traversal instead of brute-force loop
    bool hit = bvh_closest_hit(
        ray_origin,
        ray_direction,
        bvhNodes,
        leafTriIndices,
        nfaces,
        vertices,
        faces,
        tMin,
        &tBest,
        &faceIdBest,
        &uBest,
        &vBest);

    const uint idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = faceIdBest;

    float ao = 1.0f; // background stays white

    if (faceIdBest >= 0) {
        uint3  f = vload3((uint)faceIdBest, faces);
        float3 a = vload3(f.x, vertices);
        float3 b = vload3(f.y, vertices);
        float3 c = vload3(f.z, vertices);

        float3 e1 = b - a;
        float3 e2 = c - a;
        float3 n  = fast_normalize(cross(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (dot(n, ray_direction) > 0.0f) {
            n = -n;
        }

        float3 P = ray_origin + tBest * ray_direction;

        float3 ac = c - a;
        float  scale = fmax(fmax(fast_length(e1), fast_length(e2)),
                            fast_length(ac));

        float  eps = 1e-3f * fmax(1.0f, scale);
        float3 Po  = P + n * eps;

        // build tangent basis
        float3 T;
        float3 B;
        make_basis(n, &T, &B);

        // per-pixel seed (stable)
        union {
            float f32;
            uint  u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint rng = (uint)(0x9E3779B9u ^ idx ^ tBestUnion.u32);

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1  = random01(&rng);
            float u2  = random01(&rng);
            float z   = u1;                      // z in [0,1]
            float phi = 6.28318530718f * u2;     // 2*pi*u2
            float r   = native_sqrt(fmax(0.0f, 1.0f - z * z));
            float3 d_local = (float3)(r * native_cos(phi),
                                      r * native_sin(phi),
                                      z);

            // transform to world space
            float3 d = T * d_local.x + B * d_local.y + n * d_local.z;

            if (any_hit_from(Po, d,
                             vertices, faces,
                             bvhNodes, leafTriIndices,
                             nfaces, faceIdBest))
            {
                ++hits;
            }
        }

        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }

    framebuffer_ambient_occlusion[idx] = ao;
}
