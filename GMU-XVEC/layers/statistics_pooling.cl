__kernel void pool_statistics(float variance_floor, unsigned long dim, unsigned long rows, __global float *val, __global float *output) {
    int globalId = get_global_id(0);
    unsigned long cols = dim * 2 + 1;
    if (globalId >= cols) {
        return;
    }
    float cnt = val[0];

    if (globalId >= dim) {
        float mean_val = val[globalId - dim] / cnt;
        float std_val = val[globalId] / cnt - mean_val * mean_val;
        output[globalId - 1] = sqrt(std_val > variance_floor ? std_val : variance_floor);
    }
    else {
        output[globalId] = val[globalId + 1] / cnt;
    }
}
    
    
    
