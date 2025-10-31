#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    __global       uint* block_sum,
    unsigned int current_pow,
    unsigned int n)
{

    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);
    uint group_index = get_group_id(0);
    uint num_groups = get_num_groups(0);

    __local uint mem[TILE_SIZE];
    __local uint mem2[TILE_SIZE];

    mem[local_index] = pow2_sum[group_index * TILE_SIZE + local_index];
    mem2[local_index] = pow2_sum[group_index * TILE_SIZE + local_index];

    uint pow = 2;
    for (uint pow = 2; pow < TILE_SIZE + 1; pow <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        uint result_index = (local_index + 1) * pow - 1;
        uint summed_index = (local_index + 1) * pow - (pow >> 1) - 1;
        if (result_index < TILE_SIZE) {
            // printf("%u %u\n", result_index, summed_index);
            mem[result_index] += mem[summed_index];
        }
    }
    
    mem[TILE_SIZE - 1] = 0;

    uint offset = TILE_SIZE;
    for (uint d = 1; d < TILE_SIZE; d *= 2) {
     // traverse down tree & build scan
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_index < d) {
            int ai = offset * (2 * local_index + 1) - 1;
            int bi = offset * (2 * local_index + 2) - 1;
            float t = mem[ai];
            mem[ai] = mem[bi];
            mem[bi] += t;
        }
    }


    barrier(CLK_LOCAL_MEM_FENCE);
    if (group_index * TILE_SIZE + local_index < n) {
        next_pow2_sum[group_index * TILE_SIZE + local_index] = mem[local_index] + mem2[local_index];
    }
    if (local_index == 0) {
        block_sum[group_index] = mem[TILE_SIZE - 1] + mem2[TILE_SIZE - 1];
    }

}