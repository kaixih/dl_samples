DIR=$(dirname $0)
python $DIR/conv_bn.py --bench -N 128 -H 128 -W 128 -C 1024 -K 512 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 128 -W 128 -C 1024 -K 256 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 128 -W 128 -C 1024 -K 128 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 128 -W 128 -C 512 -K 1024 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 128 -W 128 -C 512 -K 512 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 128 -W 128 -C 512 -K 256 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 64 -W 64 -C 512 -K 512 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 128 -H 32 -W 32 -C 512 -K 512 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 256 -H 128 -W 128 -C 512 -K 512 -R 2 -S 2 \
    |& grep '^Results'
python $DIR/conv_bn.py --bench -N 64 -H 128 -W 128 -C 512 -K 512 -R 2 -S 2 \
    |& grep '^Results'

DIR=/home/workspace/repo_zoo/cudnn_frontend_test
$DIR/run_conv_graphs.out -graph_index 7 -input 128,1024,128,128 -filter 512,1024,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,1024,128,128 -filter 256,1024,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,1024,128,128 -filter 128,1024,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,512,128,128 -filter 1024,512,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,512,128,128 -filter 512,512,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,512,128,128 -filter 256,512,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,512,64,64 -filter 512,512,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 128,512,32,32 -filter 512,512,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 256,512,128,128 -filter 512,512,2,2 -padding 0,0 | grep '^Execution'
$DIR/run_conv_graphs.out -graph_index 7 -input 64,512,128,128 -filter 512,512,2,2 -padding 0,0 | grep '^Execution'

