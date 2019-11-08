#!/usr/bin/bash
# limit numper of OpenMP threads
# export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1



for i in {7..9}
do
    python exp_sim_pore_dist.py -p -F \
    AFFDK_results/sim_pore_dist with it_i=$i
done



