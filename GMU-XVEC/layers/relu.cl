// borrowed from https://github.com/hughperkins/DeepCL/blob/master/cl/activate.cl
kernel void activation(const int N, global float *inout) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    inout[globalId] = inout[globalId] > 0 ? inout[globalId] : 0;
}
