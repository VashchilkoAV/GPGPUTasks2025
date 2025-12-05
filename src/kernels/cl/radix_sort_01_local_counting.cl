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
    __local uint sum[NUM_BUCKETS];// = 69761289000;
    
    __local uint mem[NUM_BUCKETS][GROUP_SIZE];

    for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
        if (global_id < a2) {
            mem[bucket][local_id] = ((buffer1[global_id] & flag) == bucket) ? 1 : 0;
        } else {
            mem[bucket][local_id] = 0;
        }
    }
    
    for (uint pow = 2; pow < GROUP_SIZE + 1; pow <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
            uint result_index = (local_id + 1) * pow - 1;
            uint summed_index = (local_id + 1) * pow - (pow >> 1) - 1;
            if (result_index < GROUP_SIZE) {
                mem[bucket][result_index] += mem[bucket][summed_index];
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
            sum[bucket] = mem[bucket][GROUP_SIZE - 1];
            mem[bucket][GROUP_SIZE - 1] = 0; // set previous bucket sum
        }
    }

    uint offset = GROUP_SIZE;
    for (uint d = 1; d < GROUP_SIZE; d *= 2) {
     // traverse down tree & build scan
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
                int ai = offset * (2 * local_id + 1) - 1;
                int bi = offset * (2 * local_id + 2) - 1;
                uint t = mem[bucket][ai];
                mem[bucket][ai] = mem[bucket][bi];
                mem[bucket][bi] += t;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id < a2) {
        for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
            buffer2[a2 * bucket + global_id] = mem[bucket][local_id];
        }
    }
    if (local_id == 0) {
        for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
            buffer3[DIV_CEIL(a2, GROUP_SIZE) * bucket + group_id] = sum[bucket];
        }
    }
}