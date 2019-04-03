#!/usr/bin/bash
# limit numper of OpenMP threads
# export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1


for i in {0..8}
do
    python exp_sim_var_cone.py -p -F \
    AFFDK_results/sim_varied_cone with it_i=$i
done



