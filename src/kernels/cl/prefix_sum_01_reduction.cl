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
    unsigned int current_pow,
    unsigned int n)
{
    uint global_index = get_global_id(0);

    uint sum = 0;
    uint pow_multiplier = (uint) (1) << current_pow;
    uint pow_offset = (uint) (1) << (current_pow - 1);

    for (uint reduction_step_num = 0; reduction_step_num < NUM_REDUCTIONS_PER_RUN; reduction_step_num++) {
        if ((global_index + 1) * pow_multiplier - 1 < n) {
            rassert((global_index + 1) * pow_multiplier - 1 < n, 24572375);
            rassert((global_index + 1) * pow_multiplier - 1 - pow_offset < n, 4697045978);
            if (reduction_step_num == 0) {
                next_pow2_sum[(global_index + 1) * pow_multiplier - 1] = pow2_sum[(global_index + 1) * pow_multiplier - 1] + pow2_sum[(global_index + 1) * pow_multiplier - 1 - pow_offset];
            } else {
                next_pow2_sum[(global_index + 1) * pow_multiplier - 1] = next_pow2_sum[(global_index + 1) * pow_multiplier - 1] + next_pow2_sum[(global_index + 1) * pow_multiplier - 1 - pow_offset];
            }
        } else {
            // printf("%u\n", global_index);
        }
        
        // if (global_index == 0) {
        //     printf("%u %u %u %u\n", (global_index + 1) * pow_multiplier - 1, (global_index + 1) * pow_multiplier - 1 - pow_offset, pow2_sum[(global_index + 1) * pow_multiplier - 1], pow2_sum[(global_index + 1) * pow_multiplier - 1 - pow_offset]);
        // }
        pow_multiplier = pow_multiplier << 1;
        pow_offset = pow_offset << 1;
    }
}
