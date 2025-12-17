#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "geometry_helpers.cl"


#include "../defines.h"

uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

uint3 expandBitsVector(uint3 v) {
    // if (v != (v & 0x3FFu)) {
    //     printf("shit\n");
    // }

    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

MortonCode morton3D(float3 baricenter) {
    // scalar version
    // float x = min(max(baricenter.x * 1024.0f, 0.f), 1023.0f);
    // float y = min(max(baricenter.y * 1024.0f, 0.f), 1023.0f);
    // float z = min(max(baricenter.z * 1024.0f, 0.f), 1023.0f);

    // uint xx = expandBits((uint) x);
    // uint yy = expandBits((uint) y);
    // uint zz = expandBits((uint) z);

    // return (xx << 2) | (yy << 1) | zz;



    
    // uint xx = expandBits((uint) baricenter_clamped.x);
    // uint yy = expandBits((uint) baricenter_clamped.y);
    // uint zz = expandBits((uint) baricenter_clamped.z);

    // return (xx << 2) | (yy << 1) | zz;



    // vector version
    float3 baricenter_clamped = clamp(baricenter * 1024.f, 0.f, 1023.f);
    uint3 expanded = expandBitsVector(convert_uint3(baricenter_clamped));

    return (expanded.x << 2) | (expanded.y << 1) | expanded.z;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void ray_tracing_map_via_morton_code(
    __global const float*     vertices,
    __global const uint*      faces,
    __global uint*            mortonOut,
    float3                    minC,
    float3                    maxC,
    uint                      nFaces)
{
    const uint global_id = get_global_id(0);
    if (global_id == 0) {
        printf("%f %f %f -- %f %f %f\n", minC.x, minC.y, minC.z, maxC.x, maxC.y, maxC.z);
    }

    if (global_id < nFaces) {
        const uint3 face = loadFace(faces, global_id);

        const float3 v0 = loadVertex(vertices, face.x);
        const float3 v1 = loadVertex(vertices, face.y);
        const float3 v2 = loadVertex(vertices, face.z);

        const float3 baricenter = native_divide(v0 + v1 + v2, 3.f);
        // const float3 nBaricenter = native_divide(baricenter - minC, clamp(maxC - minC, EPS, FLT_MAX));

        //maybe
        const float3 nBaricenter = clamp(native_divide(baricenter - minC, clamp(maxC - minC, EPS, FLT_MAX)), 0.0f, 1.0f);

        const uint mortonCode = morton3D(nBaricenter);
        mortonOut[global_id] = mortonCode;

        // if (global_id < 10) {
        //     printf("%u -- %u\n", global_id, mortonCode);
        // }
    }
}