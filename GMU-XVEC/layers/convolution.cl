__kernel void convolve(__global float *input, __global float *output, __global float *weights, unsigned long input_dim, unsigned long output_dim, unsigned long input_depth, unsigned long num_filters, unsigned long kernel_dim, unsigned long stride) {
    int globalId = get_global_id(0);
    if (globalId >= output_dim * output_dim)
        return;
    unsigned long kernel_dim_half = kernel_dim / 2;
    unsigned long output_row = globalId / output_dim;
    unsigned long output_col = globalId % output_dim;
    float filter_value, input_value;
    unsigned long filter_offset;
    int row_jump, col_jump;
    for (unsigned long f = 0; f < num_filters; f++) {
        unsigned long output_offset = f * output_dim * output_dim;
        float depth_sum = 0.0f;
        for (unsigned long id = 0; id < input_depth; id++) {
            float sum = 0.0f;
            unsigned long input_offset = id * input_dim * input_dim;
            unsigned long input_row = output_row * stride + kernel_dim_half;
            unsigned long input_col = output_col * stride + kernel_dim_half;
            filter_offset = f * kernel_dim * kernel_dim * input_depth + id * kernel_dim * kernel_dim;
//            if (globalId == 1)
//                printf("###Filter offset: %d\n", filter_offset);
        
            for (unsigned long i = 0; i < kernel_dim; i++) {
                if (i == kernel_dim_half)
                    row_jump = 0;
                else
                    row_jump = i < kernel_dim_half ? -kernel_dim_half + i : kernel_dim_half;
                for (unsigned long j = 0; j < kernel_dim; j++) {
                    if (j == kernel_dim_half)
                        col_jump = 0;
                    else
                        col_jump = j < kernel_dim_half ? -kernel_dim_half + j : kernel_dim_half;
                    
                    filter_value = weights[filter_offset + i * kernel_dim + j];
                    input_value = input[(input_row + row_jump) * input_dim + (input_col + col_jump) + input_offset];
                    sum += input_value * filter_value;
//                    if (globalId == 1) {
//                        printf("%d %d %d %d %f %f\n", row_jump, col_jump, (input_row + row_jump) * input_dim, input_col + col_jump, input_value * filter_value, sum);
////                        printf("%d %d %d %d %d %d %f %f\n", input_row, input_col, row_jump, col_jump, (input_row + row_jump) * input_dim, input_col + col_jump, input_value * filter_value, sum);
//                        printf("filter_value: %f input_value: %f sum: %f\n", filter_value, input_value, sum);
//                    }
                }
            }
            depth_sum += sum;
        }
//        if (globalId == 1)
//            printf("Writing value %f to index %d\n", depth_sum, output_row * output_dim + output_col + output_offset);
        output[output_row * output_dim + output_col + output_offset] = depth_sum;
    }
}
