#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"

#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "lbvh_helpers.cl"



#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void ray_tracing_make_nodes(
    __global const uint*      mortonCodes,
    __global const float*     vertices,
    __global const uint*      faces,
    __global const uint*      triIndices,
    __global BVHNodeGPU*      nodes,
    __global uint*            parents,
    uint                      nFaces)
{
    const uint global_id = get_global_id(0);
    
    
    // make leaf node
    if (global_id < nFaces) {
        const uint faceId = triIndices[global_id];
        // const uint faceId = get_group_id
        const uint3 face = loadFace(faces, faceId);
        const float3 v0 = loadVertex(vertices, face.x);
        const float3 v1 = loadVertex(vertices, face.y);
        const float3 v2 = loadVertex(vertices, face.z);
        
        AABBGPU aabb;
        aabb.min_x = fmin(fmin(v0.x, v1.x), v2.x);
        aabb.max_x = fmax(fmax(v0.x, v1.x), v2.x);
        aabb.min_y = fmin(fmin(v0.y, v1.y), v2.y);
        aabb.max_y = fmax(fmax(v0.y, v1.y), v2.y);
        aabb.min_z = fmin(fmin(v0.z, v1.z), v2.z);
        aabb.max_z = fmax(fmax(v0.z, v1.z), v2.z);

        BVHNodeGPU node;
        node.aabb = aabb;
        node.leftChildIndex = UINT_MAX; // analog to INVALID in build_bvh_cpu.h
        node.rightChildIndex = UINT_MAX;


        nodes[nFaces - 1 + global_id] = node;
        // nodes[nFaces + global_id] = node;
    }

    // make inner node
    if (global_id < nFaces - 1) {
        int first, last;
        determine_range(mortonCodes, nFaces, global_id, &first, &last);
        int split = find_split(mortonCodes, nFaces, first, last);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = convert_int((nFaces - 1) + split);
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = convert_int((nFaces - 1) + split + 1);
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU node;
        node.leftChildIndex  = convert_uint(leftIndex);
        node.rightChildIndex = convert_uint(rightIndex);
        parents[leftIndex] = global_id;
        parents[rightIndex] = global_id;

        nodes[global_id] = node;
    }

}