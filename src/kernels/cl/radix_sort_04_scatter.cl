#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global       uint* buffer3,
    unsigned int a1,
    unsigned int a2,
    __global       uint* buffer4)
{
    const uint global_id = get_global_id(0);
    if (global_id >= a2) {
        return;
    }

    const uint flag = ((1 << BIT_PER_RUN) - 1) << a1; // ((1 << (BIT_PER_RUN + 1)) - 1) -- row of ones with length=BIT_PER_RUN
    const uint last_flag = buffer1[a2 - 1] & flag ? 0 : 1;

    const uint result_index = (buffer1[global_id] & flag) ? global_id - buffer2[global_id] + buffer2[a2 - 1] + last_flag : buffer2[global_id]; // problems with 51
   
    buffer3[result_index] = buffer1[global_id];
}