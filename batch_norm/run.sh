#!/bin/bash
set -x

nvcc cudnn_batch_norm.cu -lcudnn -o cudnn_batch_norm.out && \
  ./cudnn_batch_norm.out
