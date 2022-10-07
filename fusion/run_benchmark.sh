#!/bin/bash

FUSION_CMD_PREFIX=
NOFUSION_CMD_PREFIX=
declare -a CMDS=("python conv_bias_activation_xla_fusion_benchmark.py -x -a 1"
                 "python conv_bias_activation_xla_no_fusion_benchmark.py -x -a 1")

for CMD in "${CMDS[@]}"; do
  $CMD -i 256,64,55,55 -f 64,64,3,3 |& grep '^Results'
  $CMD -i 256,64,55,55 -f 256,64,3,3 |& grep '^Results'
  $CMD -i 256,256,55,55 -f 64,256,3,3 |& grep '^Results'
  $CMD -i 256,256,55,55 -f 256,256,3,3 |& grep '^Results'

  $CMD -i 256,256,28,28 -f 256,256,3,3 |& grep '^Results'
  $CMD -i 256,256,28,28 -f 512,256,3,3 |& grep '^Results'
  $CMD -i 256,512,28,28 -f 256,512,3,3 |& grep '^Results'
  $CMD -i 256,512,28,28 -f 512,512,3,3 |& grep '^Results'

  $CMD -i 256,512,7,7 -f 512,512,3,3 |& grep '^Results'
  $CMD -i 256,512,7,7 -f 1024,512,3,3 |& grep '^Results'
  $CMD -i 256,1024,7,7 -f 512,1024,3,3 |& grep '^Results'
  $CMD -i 256,1024,7,7 -f 1024,1024,3,3 |& grep '^Results'

  $CMD -i 256,1024,1,1 -f 1024,1024,1,1 |& grep '^Results'
  $CMD -i 256,1024,1,1 -f 2048,1024,1,1 |& grep '^Results'
  $CMD -i 256,2048,1,1 -f 1024,2048,1,1 |& grep '^Results'
  $CMD -i 256,2048,1,1 -f 2048,2048,1,1 |& grep '^Results'
done
