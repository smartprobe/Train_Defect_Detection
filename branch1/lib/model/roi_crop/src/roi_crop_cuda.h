
int BilinearSamplerBHWD_updateOutput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output);

int BilinearSamplerBHWD_updateGradInput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput);
