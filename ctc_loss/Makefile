all:
	nvcc ctc_cudnn.cu -std=c++17 -lcudnn -o ctc_cudnn_f.out
	nvcc ctc_cudnn.cu -std=c++17 -lcudnn -o ctc_cudnn_d.out -DDtype=double
	nvcc ctc_cudnn_v8.cu -std=c++17 -lcudnn -o ctc_cudnn_v8_f.out
	nvcc ctc_cudnn_v8.cu -std=c++17 -lcudnn -o ctc_cudnn_v8_d.out -DDtype=double

clean:
	rm *.out
