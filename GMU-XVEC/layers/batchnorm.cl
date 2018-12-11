
__kernel void batchnorm(const unsigned long N, const unsigned long mean_dim, __global float *mean, global float *variance, __global float *epsilon, __global float *inout) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    inout[globalId] = (inout[globalId] - mean[globalId % mean_dim]) / (sqrt(variance[globalId % mean_dim] + epsilon[0]));
//    if (globalId == 1)
//        printf("%d %d %d %f %f %f %f\n", N, globalId, mean_dim, inout[globalId], mean[globalId % mean_dim], sqrt(variance[globalId % mean_dim]), inout[globalId]);
}
