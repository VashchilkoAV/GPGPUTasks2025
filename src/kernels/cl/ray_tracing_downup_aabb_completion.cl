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
__kernel void ray_tracing_downup_aabb_completion(
    __global const uint*      parents,
    __global BVHNodeGPU*      nodes,
    __global uint*            flags,
    uint                      nFaces)
{
    const uint global_id = get_global_id(0);

    if (global_id < nFaces) {
        uint nodeId = parents[nFaces - 1 + global_id];
        while (true) {
            if (nodeId >= nFaces * 2 - 1) {
                break;
            }
            BVHNodeGPU current_node = nodes[nodeId];
            
            const uint val = atomic_inc(flags + nodeId);
            if (val == 0) {
                return;
            }

            const AABBGPU leftAABB = nodes[current_node.leftChildIndex].aabb;
            const AABBGPU rightAABB = nodes[current_node.rightChildIndex].aabb;

            AABBGPU aabb;
            aabb.min_x = min(leftAABB.min_x, rightAABB.min_x);
            aabb.max_x = max(leftAABB.max_x, rightAABB.max_x);
            aabb.min_y = min(leftAABB.min_y, rightAABB.min_y);
            aabb.max_y = max(leftAABB.max_y, rightAABB.max_y);
            aabb.min_z = min(leftAABB.min_z, rightAABB.min_z);
            aabb.max_z = max(leftAABB.max_z, rightAABB.max_z);

            current_node.aabb = aabb;
            nodes[nodeId] = current_node;

            if (nodeId == 0) {
                break;
            } else {
                nodeId = parents[nodeId];
            }
        }
    }
    

    // if (global_id < nFaces) {
    //     while (1) {
    //         if (leaf) {

    //         } else {
    //             if (both children are set) {
    //                 calculate_aabb;
    //                 break;
    //             }
    //         }
    //     }
    // }

}