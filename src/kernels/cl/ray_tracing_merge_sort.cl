#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "geometry_helpers.cl"
#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void ray_tracing_merge_sort(
    __global const uint* input_key,
    __global       uint* output_key,
    __global const uint* input_index,
    __global       uint* output_index,
                   uint  sorted_k,
                   uint  n)
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint num_comparison_pair = global_id / (sorted_k * 2);
    const uint num_subarray_in_pair = global_id / sorted_k % 2;
    const uint num_element_in_subarray = global_id % sorted_k;

    if (global_id >= n) {
        return;
    }

    uint key = input_key[global_id];
    uint index = input_index[global_id];

    const uint start_index = num_comparison_pair * (sorted_k * 2) + sorted_k * ((num_subarray_in_pair + 1) % 2);

    bool found = 0;
    uint found_subarray_index = 0;

    // printf("%u %u %u %u %u\n", global_id, num_comparison_pair, num_subarray_in_pair, num_element_in_subarray, start_index);


    // add conditions when there is no need to do a binsearch 
    // 1) we are in the left subarray and our last element is lower than first element of right subarray
    // 2) we are in the right subarray and our first element is higher than last element of left subarray
    // 3) we are in the right subarray and current element is lower than first element of right subarray
    // 4) we are in the left subarray and current element is higher than last element of left subarray
    
    // if (num_subarray_in_pair == 0 && (start_index >= n || input_key[start_index - 1] <= input_key[start_index])) {
    //     found_subarray_index = 0;
    // } else if (num_subarray_in_pair == 1 && input_key[start_index + sorted_k - 1] <= input_key[start_index + sorted_k]) {
    //     found_subarray_index = sorted_k;
    // } else if (num_subarray_in_pair == 1 && input_key[start_index] > input_key[min(start_index + sorted_k + sorted_k - 1, n - 1)]) {
    //     found_subarray_index = 0;
    // }
    // else if (num_subarray_in_pair == 0 && input_key[start_index - sorted_k] > input_key[min(start_index + sorted_k - 1, n - 1)]) {
    //     found_subarray_index = min(sorted_k, n - start_index);
    // } else {
        int l = -1, r = sorted_k;
        for (uint j = 0; j < sorted_k; j++) {
            uint m = (l + r) / 2;
            uint delim_value = start_index + m < n ? input_key[start_index + m] : INT_MAX;
            if (num_subarray_in_pair == 0) {
                if (delim_value < key) {
                    l = m;
                } else {
                    r = m;
                }
            } else {
                if (delim_value <= key) {
                    l = m;
                } else {
                    r = m;
                }
            }

            if (l >= r - 1) {
                found_subarray_index = r;
                break;
            }
        }
    // }
    
    
    uint result_index = num_comparison_pair * (sorted_k * 2) + num_element_in_subarray + found_subarray_index;

    if (result_index < n) {
        output_key[result_index] = key;
        // output_value[result_index] = value;
        // setFace(output_value, value, result_index);
        output_index[result_index] = index;
    }
}