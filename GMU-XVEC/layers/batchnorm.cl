
__kernel void batchnorm(const unsigned long N, __global float *mean, global float *variance, __global float *epsilon, __global float *inout) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    inout[globalId] = (inout[globalId] - mean[globalId]) / (sqrt(variance[globalId] + epsilon[0]));
}
