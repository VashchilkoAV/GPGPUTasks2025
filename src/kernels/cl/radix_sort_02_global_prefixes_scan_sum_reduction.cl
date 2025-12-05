#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int previous_pow_reduction,
    unsigned int n)
{
    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);
    uint group_offset = get_group_id(0) * GROUP_SIZE;

    __local uint mem[NUM_BUCKETS][GROUP_SIZE];

    uint sum = 0;
    uint pow_multiplier = 2;
    uint pow_offset = 1 << (previous_pow_reduction);

    uint pow2_sum_index = (global_index + 1) * pow_offset - 1;
    for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
        if (pow2_sum_index < n) {
            // printf("%u -- prev[%u]=%u\n", local_index, pow2_sum_index, pow2_sum[pow2_sum_index]);
            mem[bucket][local_index] = pow2_sum[n * bucket + pow2_sum_index];
        } else {
            mem[bucket][local_index] = 0;
        }
    }

    pow_offset = 1;

    for (uint reduction_step_num = 0; reduction_step_num < NUM_REDUCTIONS_PER_RUN; reduction_step_num++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
            if ((local_index + 1) * pow_multiplier - 1 < GROUP_SIZE && group_offset + (local_index + 1) * pow_multiplier - 1 < n) {
                mem[bucket][(local_index + 1) * pow_multiplier - 1] = mem[bucket][(local_index + 1) * pow_multiplier - 1] + mem[bucket][(local_index + 1) * pow_multiplier - 1 - pow_offset];
            }
        }

        pow_multiplier = pow_multiplier << 1;
        pow_offset = pow_offset << 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_index < n) {
        for (uint bucket = 0; bucket < NUM_BUCKETS; bucket++) {
            next_pow2_sum[n * bucket + global_index] = mem[bucket][local_index];
        }
    }
}
