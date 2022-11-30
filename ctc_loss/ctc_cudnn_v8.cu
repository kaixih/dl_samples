#include <iostream>
#include <cstring>
#include <iomanip>
#include "cudnn.h"

#define CUDNN_CALL(func)                                                       \
  {                                                                            \
    auto e = (func);                                                           \
    if (e != CUDNN_STATUS_SUCCESS) {                                           \
        std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__;         \
        std::cerr << " : " << cudnnGetErrorString(e) << std::endl;             \
        exit(1);                                                               \
    }                                                                          \
  }

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    auto e = (func);                                                           \
    if ((func) != cudaSuccess) {                                               \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__;          \
        std::cerr << " : " << cudaGetErrorString(e) << std::endl;              \
        exit(1);                                                               \
    }                                                                          \
  }

#ifndef Dtype
#define Dtype float
#endif

int main() {
    const int kNumTimestamps = 4;
    const int kNumLabels = 3 + 1;
    const int kBatchSize = 5;

    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    auto cudnn_dtype = CUDNN_DATA_FLOAT;
    if (std::is_same_v<Dtype, double>) cudnn_dtype = CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t probs;
    Dtype* pProbs;
    {
      CUDNN_CALL(cudnnCreateTensorDescriptor(&probs));
      const int dims[] {kNumTimestamps, kBatchSize, kNumLabels};
      const int strides[] {kBatchSize * kNumLabels, kNumLabels, 1};
      CUDNN_CALL(cudnnSetTensorNdDescriptor(probs, cudnn_dtype, 3, dims,
                                            strides));
      int total_size = kNumLabels * kNumTimestamps * kBatchSize;
      CUDA_CALL(cudaMallocManaged(&pProbs, sizeof(Dtype) * total_size));
      for(int i = 0; i < kNumTimestamps * kBatchSize; i++) {
        pProbs[i * kNumLabels + 0] = 1.f;
        pProbs[i * kNumLabels + 1] = 2.f;
        pProbs[i * kNumLabels + 2] = 3.f;
        pProbs[i * kNumLabels + 3] = 4.f;
        // pProbs[i * kNumLabels + 0] = 1.f;
        // pProbs[i * kNumLabels + 1] = 1.f;
        // pProbs[i * kNumLabels + 2] = 1.f;
        // pProbs[i * kNumLabels + 3] = 1.f;
      }
    }

    cudnnTensorDescriptor_t grads;
    Dtype* pGrads;
    {
      CUDNN_CALL(cudnnCreateTensorDescriptor(&grads));
      const int dims[] {kNumTimestamps, kBatchSize, kNumLabels};
      const int strides[] {kBatchSize * kNumLabels, kNumLabels, 1};
      CUDNN_CALL(cudnnSetTensorNdDescriptor(grads, cudnn_dtype, 3, dims,
                                            strides));
      int total_size = kNumLabels * kNumTimestamps * kBatchSize;
      CUDA_CALL(cudaMalloc(&pGrads, sizeof(Dtype) * total_size));
    }

    cudnnCTCLossDescriptor_t ctcLossDesc;
    CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctcLossDesc));
    CUDNN_CALL(cudnnSetCTCLossDescriptor_v8(
                   /*ctcLossDesc=*/ctcLossDesc,
                   /*compType=*/cudnn_dtype,
                   /*normMode=*/CUDNN_LOSS_NORMALIZATION_SOFTMAX,
                   /*gradMode=*/CUDNN_NOT_PROPAGATE_NAN,
                   /*maxLabelLength=*/3));

    size_t workspace_size;

    int labels_host[] {1, 2, 3,
                       1, 2, 3,
                       1, 2, 3,
                       1, 2, 3,
                       1, 2, 3};
    int labelLengths_host[] {3, 3, 3, 3, 3};
    int inputLengths_host[] {4, 4, 4, 4, 4};

    int* labels;
    int* labelLengths;
    int* inputLengths;
    CUDA_CALL(cudaMalloc(&labels, sizeof(int) * kBatchSize * (kNumLabels - 1)));
    CUDA_CALL(cudaMalloc(&labelLengths, sizeof(int) * kBatchSize));
    CUDA_CALL(cudaMalloc(&inputLengths, sizeof(int) * kBatchSize));
    CUDA_CALL(cudaMemcpy(labels, labels_host,
                         sizeof(int) * kBatchSize * (kNumLabels - 1),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(labelLengths, labelLengths_host,
                         sizeof(int) * kBatchSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(inputLengths, inputLengths_host,
                         sizeof(int) * kBatchSize, cudaMemcpyHostToDevice));

    CUDNN_CALL(cudnnGetCTCLossWorkspaceSize_v8(
                   /*handle=*/handle,
                   /*algo=*/CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
                   /*ctcLossDesc=*/ctcLossDesc,
                   /*probsDesc=*/probs,
                   /*gradientsDesc=*/grads,
                   /*sizeInBytes=*/&workspace_size));
    printf("Workspace in bytes: %ld\n", workspace_size);

    void *workspace;
    CUDA_CALL(cudaMalloc(&workspace, workspace_size));

    Dtype *costs;
    CUDA_CALL(cudaMalloc(&costs, sizeof(Dtype) * kBatchSize));

    CUDNN_CALL(cudnnCTCLoss_v8(
                   /*handle*/handle,
                   /*algo*/CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
                   /*ctcLossDesc*/ctcLossDesc,
                   /*probsDesc*/probs,
                   /*probs*/pProbs,
                   /*hostLabels*/labels,
                   /*hostLabelLengths*/labelLengths,
                   /*hostInputLengths*/inputLengths,
                   /*costs*/costs,
                   /*gradidentsDesc*/grads,
                   /*gradidents*/pGrads,
                   /*workspaceSizeInBytes*/workspace_size,
                   /*workspace*/workspace));

    CUDA_CALL(cudaDeviceSynchronize());

    {
      std::cout << "Loss: " << std::endl;
      Dtype* ptr = new Dtype[kBatchSize];
      CUDA_CALL(cudaMemcpy(ptr, costs, sizeof(Dtype) * kBatchSize,
                cudaMemcpyDeviceToHost));
      std::cout << std::fixed;
      std::cout << std::setprecision(8);
      for( size_t i = 0; i < kBatchSize; i++) {
        std::cout << ptr[i] << ", ";
      }
      std::cout << std::endl;
      delete[] ptr;
    }

    {
      std::cout << "Grads: " << std::endl;
      int total_size = kNumLabels * kNumTimestamps * kBatchSize;
      Dtype *ptr = new Dtype[total_size];
      CUDA_CALL(cudaMemcpy(ptr, pGrads, sizeof(Dtype) * total_size,
                           cudaMemcpyDeviceToHost));
      std::cout << std::fixed;
      std::cout << std::setprecision(8);
      for(int i = 0; i < kNumTimestamps; i++) {
        for(int j = 0; j < kBatchSize; j++) {
          for(int k = 0; k < kNumLabels; k++) {
            std::cout << ptr[i * kBatchSize * kNumLabels + j * kNumLabels + k]
                << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      delete[] ptr;
    }

    cudaFree(costs);
    cudaFree(workspace);
    cudaFree(pProbs);
    cudaFree(pGrads);

    CUDNN_CALL(cudnnDestroyCTCLossDescriptor(ctcLossDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(probs));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(grads));
    CUDNN_CALL(cudnnDestroy(handle));
}
