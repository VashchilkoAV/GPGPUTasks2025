#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"


#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void ray_tracing_downup_aabb_completion(
    __global const uint*      parents,
    volatile __global BVHNodeGPU*      nodes,
    volatile __global uint*            flags,
    uint                      nFaces)
{
    const uint global_id = get_global_id(0);

    if (global_id >= nFaces * 2 - 1) return;
    if (global_id >= nFaces) return;

    #pragma unroll
    for (__private uint nodeId = parents[nFaces - 1 + global_id]; nodeId < UINT_MAX; nodeId = parents[nodeId]) {
    // while (true) {
        if (nodeId < nFaces && atomic_inc(flags + nodeId) == 0) {
            
        } else {
            nodes[nodeId].aabb.min_x = min(nodes[nodes[nodeId].leftChildIndex].aabb.min_x, nodes[nodes[nodeId].rightChildIndex].aabb.min_x);
            nodes[nodeId].aabb.max_x = max(nodes[nodes[nodeId].leftChildIndex].aabb.max_x, nodes[nodes[nodeId].rightChildIndex].aabb.max_x);
            nodes[nodeId].aabb.min_y = min(nodes[nodes[nodeId].leftChildIndex].aabb.min_y, nodes[nodes[nodeId].rightChildIndex].aabb.min_y);
            nodes[nodeId].aabb.max_y = max(nodes[nodes[nodeId].leftChildIndex].aabb.max_y, nodes[nodes[nodeId].rightChildIndex].aabb.max_y);
            nodes[nodeId].aabb.min_z = min(nodes[nodes[nodeId].leftChildIndex].aabb.min_z, nodes[nodes[nodeId].rightChildIndex].aabb.min_z);
            nodes[nodeId].aabb.max_z = max(nodes[nodes[nodeId].leftChildIndex].aabb.max_z, nodes[nodes[nodeId].rightChildIndex].aabb.max_z);
        }
    }

}