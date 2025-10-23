#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);
    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint local_w = get_local_size(0);
    const uint local_h = get_local_size(1);

    const uint tile_x = get_group_id(0);
    const uint tile_y = get_group_id(1);

    const uint tile_count = (k + local_w - 1) / local_w;

    __local float local_dataB[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float local_dataA[GROUP_SIZE_X * GROUP_SIZE_Y];

    // double value = 0; 
    float value = 0; 
    // развернуть итерацию -- итерироваться внутри столбца второй матрицы!
    for (uint tile_inner = 0; tile_inner < tile_count; tile_inner++) {
        const uint tile_col = tile_inner * local_w + local_col;
        const uint tile_row = tile_inner * local_w + local_row;

        local_dataA[local_row * local_w + local_col] = a[row * k + tile_col];
        local_dataB[local_row * local_w + local_col] = b[tile_row * w + col];            
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint inner_num = 0; inner_num < GROUP_SIZE_X; inner_num++) {
            value += local_dataA[local_row * GROUP_SIZE_X + inner_num] * local_dataB[inner_num * GROUP_SIZE_X + local_col];
            // if use local_h instead of  GROUP_SIZE_X it is 3 times slower
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[row * w + col] = value;
    
}
