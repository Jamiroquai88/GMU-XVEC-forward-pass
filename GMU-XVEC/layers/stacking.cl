__kernel void stack(__global float *input, __global float *output, unsigned long cols, unsigned long output_cols, unsigned long output_rows, unsigned long i, int pos0, unsigned long pos1, unsigned long start_row) {
    const int globalId = get_global_id(0);
    if (globalId < cols) {
        unsigned long input_x = 0;
        unsigned long y1;

        for (unsigned long x1 = pos0; x1 < pos1; x1++) {

            y1 = i * cols + globalId;

                if (x1 >= start_row && (y1 + (x1 - start_row) * output_cols) < output_cols * output_rows) {
                    output[y1 + (x1 - start_row) * output_cols] = input[globalId + input_x * cols];
            
            input_x++;
        }
    }
}
