#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    __local uint local_data[GROUP_SIZE];

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);

    uint my_sum = 0;
    for (uint i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        if (index < n / LOAD_K_VALUES_PER_ITEM) {
            my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
        }
    }

    // local_data[local_index] = a[index];
    local_data[local_index] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE); // fast!!!

    if (local_index == 0) {
        unsigned int group_sum = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; i++) {
            group_sum += (index + i < n) ? local_data[i] : 0;
        }
        atomic_add(sum, group_sum); // very slow!!!
        // sum += group_sum;
    } 
}
