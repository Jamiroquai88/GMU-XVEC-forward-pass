__kernel void stack(__global float *input, __global float *output, unsigned long cols, unsigned long output_cols, unsigned long output_rows, unsigned long i, int pos0, unsigned long pos1, unsigned long start_row) {
    const int globalId = get_global_id(0);
    if (globalId == 0) {
        unsigned long input_x = 0;
        unsigned long input_y = 0;
        for (unsigned long x1 = pos0; x1 < pos1; x1++) {
            input_y = 0;
            for (unsigned long y1 = i * cols; y1 < (i + 1) * cols; y1++) {
                
                
                if (x1 >= start_row) {
//                    printf("%d x1 >= start_row %d\n", x1, start_row);
//                    printf("y1: %d, output_cols: %d\n", y1, output_cols);
                    output[y1 + (x1 - start_row) * output_cols] = input[input_y + input_x * cols];
//                    printf("%d %d %f %f - %d %d %d\n", y1 + (x1 - start_row) * output_cols, input_y + input_x * cols, input[input_y + input_x * cols], output[y1 + (x1 - start_row) * output_cols], x1, start_row, i);
                }
                input_y++;
            }
            input_x++;
            continue;
        }
    }
}
