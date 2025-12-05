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
    __global const uint* buffer3,
    __global       uint* buffer4,
    unsigned int a1,
    unsigned int a2,
    __global       uint* buffer5,
    __global const uint* buffer6)
{
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    if (global_id >= a2) {
        return;
    }

    const uint flag = ((1 << BIT_PER_RUN) - 1) << a1; // ((1 << (BIT_PER_RUN + 1)) - 1) -- row of ones with length=BIT_PER_RUN
    const uint group_offset = DIV_CEIL(a2, GROUP_SIZE);
    const uint bucket = (buffer1[global_id] & flag) >> a1;
    uint result_index = 0;
    for (uint seq = 0; seq < bucket; seq++) {
        // result_index += buffer3[group_offset - 1 + seq * group_offset]; // + counter[group_offset - 1 + seq * group_offset]
        result_index += buffer3[group_offset - 1 + seq * group_offset] + buffer6[group_offset - 1 + seq * group_offset];
    }
    result_index += buffer3[group_offset * bucket + group_id];
    result_index += buffer2[global_id];

    buffer4[result_index] = buffer1[global_id];
    
    
    // buffer3[global_id] = buffer1[buffer2[global_id]];
}