#!/usr/bin/bash
# limit numper of OpenMP threads
# export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1



for i in {0..1}
do
    python exp_real_data.py -p -F \
    AFFDK_results/real_data with it_i=$i
done



