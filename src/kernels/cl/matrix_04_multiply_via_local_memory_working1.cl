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
    
    // Thread identifiers
    const int col = get_local_id(0); // Local row ID (max: TS)
    const int row = get_local_id(1); // Local col ID (max: TS)
    const int globalCol = get_global_id(0); // Row ID of C (0..M)
    const int globalRow = get_global_id(1); // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float Bsub[GROUP_SIZE_X * GROUP_SIZE_Y];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = k/GROUP_SIZE_X;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledCol = GROUP_SIZE_X*t + col;
        const int tiledRow = GROUP_SIZE_X*t + row;
        Bsub[row * GROUP_SIZE_X + col] = b[tiledRow*w + globalCol];
        Asub[row * GROUP_SIZE_X + col] = a[globalRow*k + tiledCol];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int kk=0; kk<GROUP_SIZE_X; kk++) {
            acc += Asub[row * GROUP_SIZE_X + kk] * Bsub[kk * GROUP_SIZE_X + col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    c[globalRow*w + globalCol] = acc;
}
// __kernel void matrix_04_multiply_via_local_memory(
//                        __global const float* a, // rows=h x cols=k
//                        __global const float* b, // rows=k x cols=w
//                        __global       float* c, // rows=h x cols=w
//                                 unsigned int w,
//                                 unsigned int h,
//                                 unsigned int k)
// {
//     const uint col = get_global_id(0);
//     const uint row = get_global_id(1);
//     const uint local_col = get_local_id(0);
//     const uint local_row = get_local_id(1);

//     const uint local_w = get_local_size(0);
//     const uint local_h = get_local_size(1);

//     const uint tile_x = get_group_id(0);
//     const uint tile_y = get_group_id(1);

//     const uint tile_count = (k + local_w - 1) / local_w;

//     __local float local_dataB[GROUP_SIZE_X * GROUP_SIZE_Y];
//     __local float local_dataA[GROUP_SIZE_X * GROUP_SIZE_Y];

//     // double value = 0; 
//     float value = 0; 
//     // развернуть итерацию -- итерироваться внутри столбца второй матрицы!
//     for (uint tile_inner = 0; tile_inner < tile_count; tile_inner++) {
//         // if ((tile_inner * local_h + local_row) * w + (tile_x * local_w + local_col) < k * w) {
//             local_dataA[local_row * local_w + local_col] = a[(tile_y * local_h + local_row) * k + (tile_inner * local_w + local_col)];
//             local_dataB[local_row * local_w + local_col] = b[(tile_inner * local_h + local_row) * w + (tile_x * local_w + local_col)];            
//         // } else {
//         //     local_data[local_row * local_w + local_col] = 0;
//         // }
        
//         barrier(CLK_LOCAL_MEM_FENCE);

//         for (uint inner_num = 0; inner_num < local_h; inner_num++) {
//             // if ((tile_y * local_h + local_row) * w + (tile_x * local_w + inner_num) < h * k) {
//             // if ((tile_y * local_h + local_row) < h && (tile_inner * local_w + inner_num) < k) {
//                 // value += (double) (a[(tile_y * local_h + local_row) * k + (tile_inner * local_w + inner_num)] * local_data[inner_num * local_w + local_col]);
//             value += local_dataA[local_row * local_w + inner_num] * local_dataB[inner_num * local_w + local_col];
//             // }
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }

//     // c[(tile_y * local_h + local_row) * w + (tile_x * local_w + local_col)] = (float)value;
//     c[(tile_y * local_h + local_row) * w + (tile_x * local_w + local_col)] = value;
    
// }
