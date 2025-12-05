#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define BIT_PER_RUN 1
#define NUM_BUCKETS (1u << ((BIT_PER_RUN - 1)))
#define ELEMENTS_PER_WORK_ITEM 1

#define NUM_REDUCTIONS_PER_RUN 6
#define INCLUSIVE 0

#define DIV_CEIL(x,y) (((x) + (y) - 1) / (y))

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
