__kernel void pool_statistics(float variance_floor, unsigned long dim, unsigned long rows, __global float *val, __global float *output, unsigned long offset) {
    int globalId = get_global_id(0);
    if (globalId >= dim * 2) {
        return;
    }
    float cnt = val[offset];

    if (globalId >= dim) {
        float mean_val = val[offset + globalId - dim] / cnt;
        float std_val = val[offset + globalId] / cnt - mean_val * mean_val;
        output[globalId - 1] = sqrt(std_val > variance_floor ? std_val : variance_floor);
    }
    else {
        output[globalId] = val[offset + globalId + 1] / cnt;
    }
}
    
    
    
