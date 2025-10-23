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

    const uint tile_x = col / local_w;
    const uint tile_y = row / local_h;

    const uint tile_count = (k + local_w - 1) / local_w;

    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    // double value = 0; 
    float value = 0; 
    // развернуть итерацию -- итерироваться внутри столбца второй матрицы!
    for (uint tile_inner = 0; tile_inner < tile_count; tile_inner++) {
        // if ((tile_inner * local_h + local_row) * w + (tile_x * local_w + local_col) < k * w) {
            local_data[local_row * local_w + local_col] = b[(tile_inner * local_h + local_row) * w + (tile_x * local_w + local_col)];            
        // } else {
        //     local_data[local_row * local_w + local_col] = 0;
        // }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint inner_num = 0; inner_num < local_h; inner_num++) {
            // if ((tile_y * local_h + local_row) * w + (tile_x * local_w + inner_num) < h * k) {
            // if ((tile_y * local_h + local_row) < h && (tile_inner * local_w + inner_num) < k) {
                // value += (double) (a[(tile_y * local_h + local_row) * k + (tile_inner * local_w + inner_num)] * local_data[inner_num * local_w + local_col]);
            value += a[(tile_y * local_h + local_row) * k + (tile_inner * local_w + inner_num)] * local_data[inner_num * local_w + local_col];
            // }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // c[(tile_y * local_h + local_row) * w + (tile_x * local_w + local_col)] = (float)value;
    c[(tile_y * local_h + local_row) * w + (tile_x * local_w + local_col)] = value;
    
}
