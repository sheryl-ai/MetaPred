#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=4:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q ShortQ

#cd $PBS_O_WORKDIR
#Graph Convolutional Networks
declare -a num_updates=(1 2 3 4 5 6)
declare -a update_batch_size=(1 2 4 8 16)
declare -a meta_batch_size=(4 8 16)
declare -a update_lr=(1e-5 1e-4 1e-3 1e-2)
declare -a meta_lr=(0.01 0.001 0.0001)
declare -a source=("DM" "AM" "PD")
declare -a iter=(1 2 3 4 5)


# for i in "${iter[@]}"
# do
#    echo iter:
#    echo "$i"
#    # python main.py --method='rnn' --metatrain_iterations=5000 --meta_batch_size=32 --update_batch_size=8 --meta_lr=0.0001 --update_lr=1e-5 --num_updates=2 --n_total_batches=500000
#    # python main.py --method='mlp' --run_time=$i --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=16 --meta_lr=0.001 --update_lr=1e-4 --num_updates=2 --n_total_batches=100000
#    python main.py --method='cnn' --source=$i --metatrain_iterations=5000 --meta_batch_size=32 --update_batch_size=8 --meta_lr=0.001 --update_lr=1e-5 --num_updates=4 --n_total_batches=500000
# done


# mlp configuration
python main.py --method='mlp' --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=16 --meta_lr=0.001 --update_lr=1e-4 --num_updates=2 --n_total_batches=100000
# cnn configuration
# python main.py --method='cnn' --metatrain_iterations=5000 --meta_batch_size=32 --update_batch_size=8 --meta_lr=0.001 --update_lr=0.0001 --num_updates=4 --n_total_batches=500000
# rnn configuration
# python main.py --method='rnn' --metatrain_iterations=5000 --meta_batch_size=32 --update_batch_size=4 --meta_lr=0.0001 --update_lr=1e-5 --num_updates=4 --n_total_batches=500000
echo All done
