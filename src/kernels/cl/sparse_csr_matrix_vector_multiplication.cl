#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* values,
    __global const uint* cols,
    __global const uint* offsets,
    __global const uint* vector,
    __global uint* output,
    unsigned int n_elements,
    unsigned int n_rows) 
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);

    __local uint accumulator[GROUP_SIZE];
    accumulator[local_id] = 0;

    const uint start_offset = offsets[group_id];
    const uint end_offset = offsets[group_id + 1];
    const uint n_iters = DIV_CEIL(end_offset - start_offset, GROUP_SIZE);

    for (uint i = 0; i < n_iters; i++) {
        uint index = start_offset + i * GROUP_SIZE + local_id;
        if (index < n_elements && index < end_offset) {
            uint col_number = cols[index];
            accumulator[local_id] += values[index] * vector[col_number];
        }
    }


    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        uint sum = 0;
        for (uint i = 0; i < GROUP_SIZE; i++) {
            sum += accumulator[i];
        }

        output[group_id] = sum;
    }
}
