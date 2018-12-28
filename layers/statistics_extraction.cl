__kernel void extract_statistics(unsigned long cols, unsigned long rows, int variance_offset, unsigned long input_cols, __global float *input, __global float *input2, __global float *output) {
    const int globalId = get_global_id(0);
    if (globalId >= cols) {
        return;
    }
    
    // copy values from input to input2 and eventualy compute cross product
    for (unsigned int i = 0; i < rows; i++) {
        if (globalId == 0) {
            input2[i * cols] = 1;
        }
        else {
            if (variance_offset != -1 && globalId >= variance_offset)
                input2[i * cols + globalId] = input[i * input_cols + globalId - input_cols - 1] * input[i * input_cols + globalId - input_cols - 1];
            else
                input2[i * cols + globalId] = input[i * input_cols + globalId - 1];
        }
    }
    output[globalId] = input2[globalId];
    for (unsigned long i = 1; i < rows; i++) {
        output[globalId + i * cols] = input2[globalId + i * cols] + output[globalId + (i - 1) * cols];
    }
}
