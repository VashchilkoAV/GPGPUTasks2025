#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k,
                   uint  n)
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint num_comparison_pair = global_id / (sorted_k * 2);
    const uint num_subarray_in_pair = global_id / sorted_k % 2;
    const uint num_element_in_subarray = global_id % sorted_k;

    if (global_id >= n) {
        return;
    }
    uint value = input_data[global_id];
    const uint start_index = num_comparison_pair * (sorted_k * 2) + sorted_k * ((num_subarray_in_pair + 1) % 2);

    bool found = 0;
    uint found_subarray_index = 0;

    // printf("%u %u %u %u %u\n", global_id, num_comparison_pair, num_subarray_in_pair, num_element_in_subarray, start_index);

    for (uint i = 0; i < sorted_k; i++) {
        // for left
        if (num_subarray_in_pair == 0) {
            if (!found && start_index + i < n && input_data[start_index + i] < value) {
                found_subarray_index++;
            }
        } else {
            if (!found && start_index + i < n && input_data[start_index + i] <= value) {
                found_subarray_index++;
            }
        }
    }
    
    uint result_index = num_comparison_pair * (sorted_k * 2) + num_element_in_subarray + found_subarray_index;
    
    // printf("%u %u %u %u %u %u %u\n", global_id, num_comparison_pair, num_subarray_in_pair, num_element_in_subarray, start_index, found_subarray_index, result_index);

    if (result_index < n) {
        output_data[result_index] = value;
    }
}
