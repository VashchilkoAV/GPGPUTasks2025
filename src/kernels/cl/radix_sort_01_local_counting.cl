#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    __global       uint* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);

    
    const uint flag = ((1 << BIT_PER_RUN) - 1) << a1;

    __local uint mem[GROUP_SIZE];

    mem[local_id] = (buffer1[global_id] & flag) ? 0 : 1;

    for (uint pow = 2; pow < GROUP_SIZE + 1; pow <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint result_index = (local_id + 1) * pow - 1;
        uint summed_index = (local_id + 1) * pow - (pow >> 1) - 1;
        if (result_index < GROUP_SIZE) {
            // printf("%u %u\n", result_index, summed_index);
            mem[result_index] += mem[summed_index];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint sum = mem[GROUP_SIZE - 1];
    mem[GROUP_SIZE - 1] = 0;

    uint offset = GROUP_SIZE;
    for (uint d = 1; d < GROUP_SIZE; d *= 2) {
     // traverse down tree & build scan
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            int ai = offset * (2 * local_id + 1) - 1;
            int bi = offset * (2 * local_id + 2) - 1;
            uint t = mem[ai];
            mem[ai] = mem[bi];
            mem[bi] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id < a2) {
        buffer2[global_id] = mem[local_id];
    }
    if (local_id == 0) {
        buffer3[group_id] = sum;
    }
    // if (local_index == 0) {
    //     block_sum[group_index] = mem[GROUP_SIZE - 1] + mem2[GROUP_SIZE - 1];
    // }
}