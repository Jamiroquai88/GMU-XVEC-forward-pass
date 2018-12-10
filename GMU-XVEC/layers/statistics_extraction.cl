__kernel void extract_statistics(unsigned long cols, unsigned long rows, unsigned long variance_offset, __global float *input, __global float *output) {
    const int globalId = get_global_id(0);
    if (globalId >= cols) {
        return;
    }
    if (variance_offset != 0)
        if (globalId >= variance_offset)
            for (unsigned long i = 0; i < rows; i++)
            input[(globalId + i * cols)] = input[(globalId + i * cols) - variance_offset + 1] * input[(globalId + i * cols) - variance_offset + 1];

    output[globalId] = input[globalId];
    for (unsigned long i = 1; i < rows; i++) {
        output[globalId + i * cols] = input[globalId + i * cols] + output[globalId + (i - 1) * cols];
        if (globalId == 0) {
//            printf("Writing to position: %d, Reading from: %d\n", globalId + i * cols, globalId + (i - 1) * cols);
//            printf("First: %f, Second: %f, Result: %f", input[globalId + i * cols], output[globalId + i * cols], output[globalId + (i - 1) * cols]);
        }
    }
}
