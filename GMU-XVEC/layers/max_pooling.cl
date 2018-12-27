__kernel void max_pool(__global float *input, __global float *output, unsigned long filter_dim, unsigned long stride, unsigned long height, unsigned long width, unsigned long depth, unsigned long output_size) {
    int globalId = get_global_id(0);
    if (globalId >= output_size)
        return;
    unsigned long current_depth = globalId / height * width;
    unsigned long depth_offset = current_depth * height * width;
    unsigned long current_width = (globalId - depth_offset) / width;
    unsigned long input_width = current_width * stride;
    unsigned long current_height = globalId / width * depth;
    unsigned long input_height = current_height * stride;
    float max_value = input[depth_offset + current_width * width + current_height];
    unsigned long width_offset;
    float value;
    for (unsigned long i = 0; i < filter_dim; i++) {
        width_offset = depth_offset + (current_width + i) * width;
        for (unsigned long j = 0; j < filter_dim; j++) {
            value = input[width_offset + j];
            if (globalId == 1)
                printf("%d %d %d %f\n", current_depth, current_width, current_height, value);
            if (max_value < value)
                value = max_value;
        }
    }
    output[globalId] = value;
}
