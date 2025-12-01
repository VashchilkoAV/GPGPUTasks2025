#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const uint global_index = get_global_id(0);
    const uint local_index = get_local_id(0);

    const uint pow2_sum_index_offset = GROUP_SIZE * (get_group_id(0) >> pow2); // ???

    uint check_index = global_index + INCLUSIVE;
    uint global_index_corrected = check_index >> pow2;


    __local uint mem[GROUP_SIZE];
    __local uint mem_add[GROUP_SIZE];

    // if (global_index < n) {
    //     mem[local_index] = prefix_sum_accum[global_index];
    // } else {
    //     mem[local_index] = 0;
    // }

    mem[local_index] = 0;

    mem_add[local_index] = pow2_sum[pow2_sum_index_offset + local_index];
    // printf("%u\n", pow2_sum[pow2_sum_index_offset + local_index]);

    uint num_reduction_steps = NUM_REDUCTIONS_PER_RUN;
    uint flag = 2;
    if (pow2 == 0) {
        flag = 1;
        num_reduction_steps++;
    }


    for (uint reduction_step_num = 0; reduction_step_num < num_reduction_steps; reduction_step_num++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (global_index < n && (global_index_corrected & flag)) {
            uint add_index = (global_index_corrected - (global_index_corrected % flag) - 1) % GROUP_SIZE;
            // if (local_index == 2)
            // printf("step=%u, local_index=%u, global_corr=%u, flag=%u, add_index=%u\n", reduction_step_num, local_index, global_index_corrected, flag, add_index);
            mem[local_index] += mem_add[add_index];
        }
        flag <<= 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_index < n) {
        prefix_sum_accum[global_index] += mem[local_index];
    }

}
