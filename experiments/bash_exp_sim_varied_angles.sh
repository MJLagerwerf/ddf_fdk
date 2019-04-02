#!/usr/bin/bash
# limit numper of OpenMP threads
# export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1


for i in {0..19}
do
    python exp_sim_varied_angles_redo.py -p -F \
    AFFDK_results/sim_varied_angles with it_i=$i
done


# python exp_sim_varied_angles.py -p -F \