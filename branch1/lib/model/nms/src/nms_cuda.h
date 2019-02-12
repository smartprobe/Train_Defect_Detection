
int nms_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
             THCudaIntTensor *num_out, float nms_overlap_thresh);
