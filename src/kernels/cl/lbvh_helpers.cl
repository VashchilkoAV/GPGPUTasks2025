#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


static inline int common_prefix(__global const uint* codes, 
                                const int nFaces, 
                                const int i, 
                                const int j)
{
    if (j < 0 || j >= nFaces) return -1;

    uint ci = codes[convert_uint(i)];
    uint cj = codes[convert_uint(j)];

    if (ci == cj) {
        uint di = convert_uint(i);
        uint dj = convert_uint(j);
        uint diff = di ^ dj;
        // return 32 + clz32(diff);
        return 32 + clz(diff);
    } else {
        uint diff = ci ^ cj;
        // return clz32(diff);
        return clz(diff);
    }
}


static inline void determine_range(__global const uint* codes, 
                                    const int nFaces, 
                                    const int i, 
                                    __private int* outFirst, 
                                    __private int* outLast)
{
    int cpL = common_prefix(codes, nFaces, i, i - 1);
    int cpR = common_prefix(codes, nFaces, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, nFaces, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, nFaces, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, nFaces, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    *outFirst = min(i, j);
    *outLast  = max(i, j);
}


static inline int find_split(__global const uint* codes,
                             const int nFaces, 
                             const int first, 
                             const int last)
{

    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = common_prefix(codes, nFaces, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, nFaces, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}