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
    unsigned int previous_pow_reduction,
    unsigned int n)
{
    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);
    uint group_offset = get_group_id(0) * GROUP_SIZE;

    __local uint mem[GROUP_SIZE];

    uint sum = 0;
    uint pow_multiplier = 2;
    uint pow_offset = 1 << (previous_pow_reduction);

    uint pow2_sum_index = (global_index + 1) * pow_offset - 1;
    if (pow2_sum_index < n) {
        // printf("%u -- prev[%u]=%u\n", local_index, pow2_sum_index, pow2_sum[pow2_sum_index]);
        mem[local_index] = pow2_sum[pow2_sum_index];
    } else {
        mem[local_index] = 0;
    }

    pow_offset = 1;

    for (uint reduction_step_num = 0; reduction_step_num < NUM_REDUCTIONS_PER_RUN; reduction_step_num++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_offset + (local_index + 1) * pow_multiplier - 1 < n) {
            mem[(local_index + 1) * pow_multiplier - 1] = mem[(local_index + 1) * pow_multiplier - 1] + mem[(local_index + 1) * pow_multiplier - 1 - pow_offset];
        }

        pow_multiplier = pow_multiplier << 1;
        pow_offset = pow_offset << 1;
    }

    next_pow2_sum[global_index] = mem[local_index];


}
