#!/bin/bash
nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum kmeans_parallel_sorting_stream ../mnist/mnist_encoded/encoded_train_ae.npy
