__kernel void max_pool(__global float *input, __global float *output, unsigned long filter_dim, unsigned long stride, unsigned long height, unsigned long width, unsigned long depth, unsigned long output_size) {
    int globalId = get_global_id(0);
    if (globalId >= output_size)
        return;
    unsigned long current_depth = (globalId * stride) / (height * width);
    unsigned long depth_offset = current_depth * height * width;
    unsigned long current_width = (((globalId * stride) - depth_offset) / width) * stride;
    unsigned long current_height = ((globalId * stride) - depth_offset) % width;
    unsigned long offset = depth_offset + current_width * width + current_height;
    float max_value = input[offset];
    float value;
    unsigned long width_offset;
    for (unsigned long i = 0; i < filter_dim; i++) {
        width_offset = offset + i * width;
        for (unsigned long j = 0; j < filter_dim; j++) {
            value = input[width_offset + j];
            if (max_value < value)
                max_value = value;
        }
    }
    output[globalId] = max_value;
}
