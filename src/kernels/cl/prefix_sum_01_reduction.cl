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

///////////////// Точно будет проблема с нечетными степенями и четным числом редукций!!!!!!!
    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);
    uint group_index = get_group_id(0);
    uint num_groups = get_num_groups(0);
    // uint group_lower_bound = group_index * ((n + num_groups - 1) / num_groups), group_upper_bound = (group_index + 1) * ((n + num_groups - 1) / num_groups);
    uint group_lower_bound = group_index * GROUP_SIZE * (uint) 1 << current_pow, group_upper_bound = (group_index + 1) * GROUP_SIZE * (uint) 1 << current_pow;
    
    group_lower_bound = 0;
    group_upper_bound = n;
    
    if (num_groups == 1) {
        group_lower_bound = 0;
        group_upper_bound = n;
    }

    for (uint reduction_step_num = 0; reduction_step_num < NUM_REDUCTIONS_PER_RUN; reduction_step_num++) {
        // if (local_index == 0 && group_index == num_groups - 1) {
        //     // printf("1 << %u = %u\n", current_pow + reduction_step_num, 1u << (current_pow + reduction_step_num));
        //     printf("%u %u\n", 8 + current_pow, num_groups);
        // }
        uint pow_multiplier = 1u << (reduction_step_num + 1);
        uint pow_offset = 1u << (reduction_step_num);
        
        // uint element_index = (global_index + 1) * pow_multiplier - 1;
        
        // remember!!!
        // uint element_index = (local_index + 1) * pow_multiplier - 1;
        // uint element_index_write = (local_index + 1) * (pow_multiplier >> 1) - 1;
        uint element_index = (global_index + 1) * pow_multiplier - 1;
        uint element_index_write = (global_index + 1) * (pow_multiplier >> 1) - 1;

        if (element_index < n) {
        // if (1) {

            rassert(element_index < n, 24572375);
            rassert(element_index - pow_offset < n, 4697045978);
            if (element_index < group_upper_bound && element_index >= group_lower_bound) {
                if (reduction_step_num == 0) {
                    // if (element_index == 515) {
                    //     printf("write to idx=%u: next[%u]=%u + next[%u]=%u\n", element_index, element_index, pow2_sum[element_index], element_index - pow_offset, pow2_sum[element_index - pow_offset]);
                    // }
                    // printf("id={%u}: el_id=%u, el-p=%u\n", global_index, element_index, element_index - pow_offset);
                    next_pow2_sum[element_index_write] = pow2_sum[element_index] + pow2_sum[element_index - pow_offset];
                    // if (element_index == 515) {
                    //     printf("write to idx=515 done, value=%u\n", next_pow2_sum[515]);
                    // }
                } else {
                    // if (element_index == 515) {
                    // // if (element_index > 511) {
                    //     // printf("write to idx=%u: next[%u]=%u + next[%u]=%u\n", element_index, element_index, next_pow2_sum[element_index], element_index - pow_offset, next_pow2_sum[element_index - pow_offset]);
                    // }
                    next_pow2_sum[element_index] = next_pow2_sum[element_index] + next_pow2_sum[element_index - pow_offset];
                }
            }
            
            
            // if (global_index == 0) {
            //     printf("next[867]=%u\n", next_pow2_sum[867]);
            // }
        } else {
            // if (reduction_step_num == 0) {
            //     printf("%u\n", element_index);
            // }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        
        // if (global_index == 0) {
        //     printf("%u %u %u %u\n", element_index, element_index - pow_offset, pow2_sum[element_index], pow2_sum[element_index - pow_offset]);
        // }
        pow_multiplier = pow_multiplier << 1;
        pow_offset = pow_offset << 1;
    }
}
