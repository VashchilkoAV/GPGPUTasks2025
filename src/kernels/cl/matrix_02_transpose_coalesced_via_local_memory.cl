#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);
    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint local_w = get_local_size(0);
    const uint local_h = get_local_size(1);

    const uint tile_x = col / local_w;
    const uint tile_y = row / local_h;

    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    // local data
    // local_data[local_row * local_w + local_col] = matrix[row * w + col];
    local_data[local_row * local_w + local_col] = matrix[(tile_y * local_h + local_row) * w + (tile_x * local_w + local_col)];
    
    
    // if (col != tile_x * local_w + local_col) {
    //     printf("col=%u, tile_col=%u, equal=%u\n", col, tile_x * local_w + local_col, col == tile_x * local_w + local_col);
    // }
   
   
   
   
    barrier(CLK_LOCAL_MEM_FENCE);

    // works! but inefficient memory
    // naive -- works
    // transposed_matrix[col * h + row] = matrix[row * w + col];

    // naive in tile notation -- works
    // transposed_matrix[(tile_x * local_w + local_col) * h + (tile_y * local_w + local_row)] = matrix[(tile_y * local_h + local_row) * w + (tile_x * local_w + local_col)];
    
    // coalesced write, uncoalesced read -- works
    // transposed_matrix[(tile_x * local_w + local_row) * h + (tile_y * local_w + local_col)] = matrix[(tile_y * local_h + local_col) * w + (tile_x * local_w + local_row)];
    
    // local memory uncoalesced write
    // transposed_matrix[(tile_x * local_w + local_col) * h + (tile_y * local_w + local_row)] = local_data[local_row * local_w + local_col];
    
    // local memory coalesced write
    transposed_matrix[(tile_x * local_w + local_row) * h + (tile_y * local_w + local_col)] = local_data[local_col * local_h + local_row];

    // if (tile_x * local_h + local_col < w && tile_y * local_w + local_row < h) {
    //     transposed_matrix[(tile_x * local_h + local_col) * h + (tile_y * local_w + local_row)] = local_data[local_col * local_w + local_row];
    // }

    // if (local_row == 0 && local_col == 0) {
    //     printf("global %u-%u, local %u-%u, global_size %u-%u, local_size %u-%u", col, row, local_col, local_row, w, h, local_w, local_h);
    // }
}
