#include <iostream>
#include <numeric>
#include <thrust/sort.h>

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

class IndexComparator {
public:
  IndexComparator(const int* indices, const int ndims) :
      indices_(indices), ndims_(ndims) {}
  inline bool operator()(const int i, const int j) const {
    for (int di = 0; di < ndims_; ++di) {
      if (indices_[i * ndims_ + di] < indices_[j * ndims_ + di]) {
        return true;
      }
      if (indices_[i * ndims_ + di] > indices_[j * ndims_ + di]) {
        return false;
      }
    }
    return false;
  }
private:
  const int *indices_;
  const int ndims_;
};

int main() {

  int ndims = 3;
  const int N = 4;

  int *indices;
  int indices_size = N * ndims;
  int indices_size_bytes = indices_size * sizeof(int);
  checkCUDA(cudaMallocManaged(&indices, indices_size_bytes));
	indices[0] = 0; indices[1]  = 3; indices[2]  = 0;
  indices[3] = 0; indices[4]  = 2; indices[5]  = 1;
  indices[6] = 1; indices[7]  = 1; indices[8]  = 0;
  indices[9] = 1; indices[10] = 0; indices[11] = 0;

  char *values;
  char values_size = N;
  char values_size_bytes = values_size * sizeof(char);
  checkCUDA(cudaMallocManaged(&values, values_size_bytes));
	values[0] = 'b'; values[1]  = 'a'; values[2]  = 'd';
	values[3] = 'c';

  int *reorder;
  int reorder_size = N;
  int reorder_size_bytes = reorder_size * sizeof(int);
  checkCUDA(cudaMallocManaged(&reorder, reorder_size_bytes));
  std::iota(reorder, reorder + N, 0);
  
  IndexComparator sorter(indices, ndims);
  thrust::sort(reorder, reorder + N, sorter);
	checkCUDA(cudaDeviceSynchronize());

  printf("permuted reorder: \n");
  for (int i = 0; i < N; i++) {
    printf("%d, ", reorder[i]);
  }
  printf("\n");

  printf("reordered entries: \n");
  for (int i = 0; i < N; i++) {
    printf("( ");
    for (int j = 0; j < ndims; j++) {
      printf("%d, ", indices[reorder[i] * ndims + j]);
    }
    printf(") -> %c\n", values[reorder[i]]);
  }
  printf("\n");

}
