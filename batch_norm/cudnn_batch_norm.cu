#include <iostream>
#include <cudnn.h>

#define checkCUDNN(expression)                             \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

void print_array(float *array, int size, const char *name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char const *argv[]) {
  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
	auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
	const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
	float one = 1.0;
  float zero = 0.0;
  int N = 2, C = 3, H = 1, W = 2;

  int x_size = N * C * H * W;
  int x_size_bytes = x_size * sizeof(float);

  int mean_size = C;
  int mean_size_bytes = mean_size * sizeof(float);

  cudnnTensorDescriptor_t x_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&x_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(x_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));
  float *x, *y, *dy, *dx;
  checkCUDA(cudaMallocManaged(&x, x_size_bytes));
  checkCUDA(cudaMallocManaged(&y, x_size_bytes));
  checkCUDA(cudaMallocManaged(&dy, x_size_bytes));
  checkCUDA(cudaMallocManaged(&dx, x_size_bytes));
  x[0]  = 0.16513085; x[2]  = 0.9014813;  x[4]  = 0.6309742;
  x[1]  = 0.4345461;  x[3]  = 0.29193902; x[5]  = 0.64250207;
  x[6]  = 0.9757855;  x[8]  = 0.43509948; x[10] = 0.6601019;
  x[7]  = 0.60489583; x[9]  = 0.6366315;  x[11] = 0.6144488;

  dy[0]  = 1.0; dy[2]  = 1.0;  dy[4]  = 1.0;
  dy[1]  = 1.0; dy[3]  = 1.0;  dy[5]  = 1.0;
  dy[6]  = 1.0; dy[8]  = 1.0;  dy[10] = 1.0;
  dy[7]  = 1.0; dy[9]  = 1.0;  dy[11] = 1.0;

  cudnnTensorDescriptor_t mean_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&mean_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(mean_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/C,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  float *scale, *offset, *dscale, *doffset;
  float *running_mean, *running_var;
  float *saved_mean, *saved_inv_var;
  checkCUDA(cudaMallocManaged(&scale, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&offset, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&dscale, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&doffset, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&running_mean, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&running_var, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&saved_mean, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&saved_inv_var, mean_size_bytes));
  // saved_mean and saved_inv_var can be nullptr.
  // saved_mean = nullptr; saved_inv_var = nullptr;

  scale[0]  = 1.0; scale[1]  = 1.0;  scale[2]  = 1.0;
  offset[0] = 0.0; offset[1] = 0.0;  offset[2] = 0.0;

  running_mean[0] = 1.0; running_mean[1] = 1.0;  running_mean[2] = 1.0;
  running_var[0]  = 1.0; running_var[1]  = 1.0;  running_var[2]  = 1.0;

  cudnnActivationDescriptor_t activation_desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc));
	checkCUDNN(cudnnSetActivationDescriptor(activation_desc,
                                          CUDNN_ACTIVATION_IDENTITY,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  size_t workspace_size_bytes = 0;
  checkCUDNN(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      /*handle=*/cudnn, /*mode=*/mode, /*bnOps=*/bn_ops,
      /*xDesc=*/x_descriptor, /*zDesc=*/NULL, /*yDesc=*/x_descriptor,
      /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
      /*activationDesc=*/activation_desc,
      /*sizeInBytes=*/&workspace_size_bytes));
  void *workspace = nullptr;
  if (workspace_size_bytes > 0) {
    checkCUDA(cudaMalloc(&workspace, workspace_size_bytes));
  }

	size_t reserve_space_size_bytes = 0;
  checkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      /*handle=*/cudnn, /*mode=*/mode, /*bnOps=*/bn_ops,
      /*activationDesc=*/activation_desc, /*xDesc=*/x_descriptor,
      /*sizeInBytes=*/&reserve_space_size_bytes));
  char *reserve_space;
  checkCUDA(cudaMalloc(&reserve_space, reserve_space_size_bytes));

  checkCUDNN(cudnnBatchNormalizationForwardTrainingEx(
             /*handle=*/cudnn,
             /*mode=*/mode,
             /*bnOps=*/bn_ops,
             /*alpha=*/&one,
             /*beta=*/&zero,
             /*xDesc=*/x_descriptor,
             /*xData=*/x,
             /*zDesc=*/NULL,
             /*zData=*/NULL,
             /*yDesc=*/x_descriptor,
             /*yData=*/y,
             /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
             /*bnScale=*/scale,
             /*bnBias=*/offset,
             /*exponentialAverageFactor=*/0.5,
             /*resultRunningMean=*/running_mean,
             /*resultRunningVariance=*/running_var,
             /*epsilon=*/0.001,
             /*resultSaveMean=*/saved_mean,
             /*resultSaveInvVariance=*/saved_inv_var,
             /*activationDesc=*/activation_desc,
             /*workspace=*/workspace,
             /*workSpaceSizeInBytes=*/workspace_size_bytes,
             /*reserveSpace=*/reserve_space,
             /*reserveSpaceSizeInBytes=*/reserve_space_size_bytes));

  checkCUDA(cudaDeviceSynchronize());

  print_array(y, x_size, "y NCHW format: ");

  checkCUDNN(cudnnBatchNormalizationBackwardEx(
      /*handle=*/cudnn,
      /*mode=*/mode,
      /*bnOps=*/bn_ops,
      /*alphaDataDiff=*/&one,
      /*betaDataDiff=*/&zero,
      /*alphaParamDiff=*/&one,
      /*betaParamDiff=*/&zero,
      /*xDesc=*/x_descriptor,
      /*xData=*/x,
      /*yDesc=*/nullptr,
      /*yData=*/nullptr,
      /*dyDesc=*/x_descriptor,
      /*dyData=*/dy,
      /*dzDesc=*/nullptr,
      /*dzData=*/nullptr,
      /*dxDesc=*/x_descriptor,
      /*dxData=*/dx,
      /*dBnScaleBiasDesc=*/mean_descriptor,
      /*bnScaleData=*/scale,
      /*bnBiasData=*/nullptr,
      /*dBnScaleData=*/dscale,
      /*dBnBiasData=*/doffset,
      /*epsilon=*/0.001,
      /*savedMean=*/saved_mean,
      /*savedInvVariance=*/saved_inv_var,
      /*activationDesc=*/activation_desc,
      /*workspace=*/workspace,
      /*workSpaceSizeInBytes=*/workspace_size_bytes,
      /*reserveSpace=*/reserve_space,
      /*reserveSpaceSizeInBytes=*/reserve_space_size_bytes));

  checkCUDA(cudaDeviceSynchronize());

  print_array(dx, x_size, "dx NCHW format: ");
  print_array(dscale, mean_size, "dscale: ");
  print_array(doffset, mean_size, "doffset: ");

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(dy));
  checkCUDA(cudaFree(dx));
  checkCUDA(cudaFree(scale));
  checkCUDA(cudaFree(offset));
  checkCUDA(cudaFree(dscale));
  checkCUDA(cudaFree(doffset));
  checkCUDA(cudaFree(running_mean));
  checkCUDA(cudaFree(running_var));
  checkCUDA(cudaFree(saved_mean));
  checkCUDA(cudaFree(saved_inv_var));
  checkCUDA(cudaFree(workspace));
  checkCUDA(cudaFree(reserve_space));
}
