#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const uint index = get_global_id(0);

    for (uint reduction_step_num = 0; reduction_step_num < NUM_REDUCTIONS_PER_RUN; reduction_step_num++) {
        const uint flag = (uint) 1 << (pow2 + reduction_step_num);
        // printf("flag=%u\n", flag);

        uint check_index = index + INCLUSIVE;
        uint power_index = check_index - (check_index % flag) - 1;


        if (index < n && (check_index & flag)) {
            prefix_sum_accum[index] += pow2_sum[power_index];
            // printf("index=%u &flag=%u powe_index=%u\n", index, check_index & flag, power_index);
        }

        if (pow2 == 0) {
            break;
        }
    }
    



    // uint sum = 0

    // for (uint flag = 1; flag < n; flag *= 2) {
    //     if (flag & index) {
    //         sum += pow2_sum[index - (index % flag)];
    //     }
    // }

    // for (uint flag1 = 1, flag2 = -1; flag1 < n; flag1 *= 2, flag2 *= 2) {
    //     if (flag1 & index) {
    //         sum += pow2_sum[flag2 & index];
    //     }
    // }


    // uint sum_index = index;
    // uint flag = -1;
    // for (uint pos = 0; pos < 32; pos++) {
    //     if (sum_index % 2) {
    //         sum += pow2_sum[(index & flag) - 1];
    //     }
    //     flag << = 1;
    // }
    
    // prefix_sum_accum[index] = sum;
}
