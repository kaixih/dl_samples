#include<iostream>
#include<algorithm>
#include<numeric>

class IndexComparator {
public:
  IndexComparator(const int *indices, const int ndims) :
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

  int indices[] = {0, 3, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0};
  char values[] = {'b', 'a', 'd', 'c'};

  IndexComparator sorter(indices, ndims);
  int reorder[N];
  std::iota(reorder, reorder + N, 0);
  std::sort(reorder, reorder + N, sorter);
    
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
